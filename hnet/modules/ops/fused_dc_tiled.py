import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()
torch.set_float32_matmul_precision('high')

"""

Fused dynamic chunking kernel based off Triton's fused softmax tutorial.

"""

def get_cuda_autotune_config():
    return [
        triton.Config({}, num_stages=2, num_warps=16),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=5, num_warps=2),
        triton.Config({}, num_stages=6, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=5, num_warps=4)
    ]

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

@triton.jit
def fused_dc_kernel(Q_ptr, # B x (L - 1) x D
            K_ptr, # B x (L - 1) x D
            p_ptr, # B x L - 1
            b_ptr, # B x L - 1
            sp_ptr,
            Q_batch_stride,
            K_batch_stride,
            p_batch_stride,
            b_batch_stride,
            sp_batch_stride,
            Q_row_stride,
            K_row_stride,
            p_row_stride,
            b_row_stride,
            sp_row_stride,
            batch_dim, # add back when we use over multiple batches
            seq_len,
            head_dim,
            ROW_PER_BLOCK: tl.constexpr,
            BLOCK_SIZE: tl.constexpr,
            num_stages: tl.constexpr
            ):

    batch_start = tl.program_id(0) # grid x axis handles a single sequence in the batch
    row_start = tl.program_id(1) # assume singular batch dim for now
    row_step = tl.num_programs(1) # grid y axis handles an entire sequence in the batch

    # get starting pointer for this element in the batch
    Q_batch_ptr = Q_ptr + batch_start * Q_batch_stride
    K_batch_ptr = K_ptr + batch_start * K_batch_stride
    p_batch_ptr = p_ptr + batch_start * p_batch_stride
    b_batch_ptr = b_ptr + batch_start * b_batch_stride
    sp_batch_ptr = sp_ptr + batch_start * sp_batch_stride

    # BOS token set as mandatory boundary for each sequence in the batch
    if row_start == 0:
        tl.store(p_batch_ptr + 0, 0.0) # p no boundary
        tl.store(p_batch_ptr + 1, 1.0) # p boundary     
        tl.store(b_batch_ptr, True)
        tl.store(sp_batch_ptr, 1.0)

    # number of row-blocks needed to cover the sequence
    num_blocks = (seq_len + ROW_PER_BLOCK - 1) // ROW_PER_BLOCK

    # instead of load row -> make unit vector -> back to global memory -> load normed row -> dot prod -> back to global memory -> get probabilities -> get boundaries
    # we want load row -> make unit vector -> dot prod -> get probabilities -> get boundaries -> back to global memory -> load clientside

    for block_id in tl.range(row_start, num_blocks, row_step, num_stages=num_stages):
        # compute row indices for this block
        row_idx     = tl.arange(0, ROW_PER_BLOCK)
        row_offsets = block_id * ROW_PER_BLOCK + row_idx
        row_mask    = row_offsets < seq_len
        safe_row_offsets = tl.where(row_mask, row_offsets, 0)

        # compute column offsets
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask    = col_offsets < head_dim

        # generate 2D pointer grids for Q and K
        Q_ptrs = Q_batch_ptr + (safe_row_offsets[:, None] * Q_row_stride) + col_offsets[None, :]
        K_ptrs = K_batch_ptr + (safe_row_offsets[:, None] * K_row_stride) + col_offsets[None, :]

        # load tiles
        mask2d = row_mask[:, None] & col_mask[None, :]
        Q_tile = tl.load(Q_ptrs, mask=mask2d, other=0.0).to(tl.float32)
        K_tile = tl.load(K_ptrs, mask=mask2d, other=0.0).to(tl.float32)

        # compute per-row dot and norms
        dot_rows  = tl.sum(Q_tile * K_tile, axis=1)
        q_norm_sq = tl.sum(Q_tile * Q_tile, axis=1)
        k_norm_sq = tl.sum(K_tile * K_tile, axis=1)
        cos_sim   = tl.fdiv(dot_rows, (tl.sqrt(q_norm_sq) * tl.sqrt(k_norm_sq)))

        # compute final values
        p_boundary = tl.clamp((1 - cos_sim) / 2, 0.0, 1.0)
        p_noboundary = 1 - p_boundary
        p = tl.join(p_noboundary, p_boundary)
        b_vals = tl.where(p_boundary >= 0.5, True, False)
        sp = tl.maximum(p_noboundary, p_boundary)

        # compute store memory addresses
        out_rows = row_offsets + 1
        safe_out_rows = tl.where(row_mask, out_rows, 0)
        
        # column offsets for two entries
        col_idxs = tl.arange(0, 2)[None, :]
        # row indices broadcasted across columns
        row_idxs = out_rows[:, None]
        # compute pointer grid for p
        p_ptrs  = p_batch_ptr + safe_out_rows[:, None] * p_row_stride + col_idxs
        b_ptrs  = b_batch_ptr + safe_out_rows * b_row_stride
        sp_ptrs = sp_batch_ptr + safe_out_rows * sp_row_stride

        # store results with proper bounds checking
        # Create 2D mask for p tensor to match its [ROW_PER_BLOCK, 2] shape
        p_mask = row_mask[:, None]
        tl.store(p_ptrs, p, mask=p_mask)
        tl.store(b_ptrs, b_vals, mask=row_mask)
        tl.store(sp_ptrs, sp, mask=row_mask)
        

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# we're ignoring batch length right now
def fused_dc(Q, K):
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Q.dim() == 3, "Q and K must be 3D tensors"
    
    batch_size, seq_len, head_dim = Q.shape

    BLOCK_SIZE = triton.next_power_of_2(head_dim)

    num_warps = 8
    num_stages = 4

    # FIX: Ensure ROW_PER_BLOCK doesn't cause overflow
    ROW_PER_BLOCK = min(num_warps // 2, 32)  # Cap at reasonable value
    num_blocks = (seq_len + ROW_PER_BLOCK - 1) // ROW_PER_BLOCK

    p = torch.empty((batch_size, seq_len + 1, 2), device=DEVICE, dtype=torch.bfloat16)
    b = torch.empty((batch_size, seq_len + 1), device=DEVICE, dtype=torch.bool)
    sp = torch.empty((batch_size, seq_len + 1, 1), device=DEVICE, dtype=torch.bfloat16)

    # Create a number of persistent programs.
    fused_dc_kernel[(batch_size, num_blocks, 1)](
        Q, K, p, b, sp,
        Q.stride(0), K.stride(0), p.stride(0), b.stride(0), sp.stride(0),
        Q.stride(1), K.stride(1), p.stride(1), b.stride(1), sp.stride(1),
        batch_size, seq_len, head_dim,
        ROW_PER_BLOCK,
        BLOCK_SIZE,
        num_stages,
    )
    return p, b, sp

if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(4, 8192, 1024, device=DEVICE, dtype=torch.bfloat16)
    K = torch.randn(4, 8192, 1024, device=DEVICE, dtype=torch.bfloat16)

    p_triton, b_triton, sp_triton = fused_dc(Q, K)

    # detach
    p = p_triton.detach().cpu()
    b = b_triton.detach().cpu()
    sp = sp_triton.detach().cpu()
