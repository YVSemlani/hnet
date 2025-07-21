import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()

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
def fused_dc_kernel_tiled(
        Q_ptr, K_ptr, p_ptr, b_ptr, sp_ptr,
        Q_batch_stride, K_batch_stride, p_batch_stride, b_batch_stride, sp_batch_stride,
        Q_row_stride, K_row_stride, p_row_stride, b_row_stride, sp_row_stride,
        batch_dim, seq_len, head_dim, # Q and K shapes
        BLOCK_SIZE: tl.constexpr,
        num_stages: tl.constexpr # pipeline stages
):
    # each block handles BLOCK_SIZE_SD tokens in the sequence
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1) * BLOCK_SIZE # iterate over the sequence dim in steps of BLOCK_SIZE_SD

    # get the starting pointer for this element in the batch
    Q_batch_ptr = Q_ptr + batch_idx * Q_batch_stride
    K_batch_ptr = K_ptr + batch_idx * K_batch_stride
    p_batch_ptr = p_ptr + batch_idx * p_batch_stride
    b_batch_ptr = b_ptr + batch_idx * b_batch_stride
    sp_batch_ptr = sp_ptr + batch_idx * sp_batch_stride

    # set BOS token to mandatory boundary
    if row_idx == 0:
        tl.store(p_batch_ptr, 0.0)
        tl.store(p_batch_ptr + 1, 1.0)
        tl.store(b_batch_ptr, True)
        tl.store(sp_batch_ptr, 1.0)

    # get the starting point for each row we're processing
    row_offsets = row_idx + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < seq_len
    Q_tile_ptr = Q_batch_ptr + row_offsets[:, None] * Q_row_stride
    K_tile_ptr = K_batch_ptr + row_offsets[:, None] * K_row_stride

    # setup accumulator
    QKt = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.bfloat16)
    Qnorm = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    Knorm = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # iterate over the head dimension in blocks of BLOCK_SIZE_HD
    for head_idx in tl.arange(0, head_dim, BLOCK_SIZE, num_stages=num_stages):
        # compute the head dimension offsets
        head_offsets = head_idx + tl.arange(0, BLOCK_SIZE)
        head_mask = head_offsets < head_dim

        # create 2D pointer grid for Q and K by advancing by head offsets along the head dim of each row
        Q_ptrs = Q_tile_ptr + (head_offsets[:, None])
        K_ptrs = K_tile_ptr + (head_offsets[:, None])
        
        # load tiles
        mask2d = row_mask[:, None] & head_mask[None, :]
        Q_tile = tl.load(Q_ptrs, mask=mask2d, other=0.0)
        K_tile = tl.load(K_ptrs, mask=mask2d, other=0.0)

        # update accumulators
        QKt += tl.dot(Q_tile, K_tile.T)
        Qnorm += tl.dot(Q_tile, Q_tile.T)
        Knorm += tl.dot(K_tile, K_tile.T)
    
    # mask out non-diagonal elements of accumulators
    QKt = tl.where(tl.arange(0, BLOCK_SIZE)[None, :] == tl.arange(0, BLOCK_SIZE)[:, None], QKt, 0.0)
    Qnorm = tl.where(tl.arange(0, BLOCK_SIZE)[None, :] == tl.arange(0, BLOCK_SIZE)[:, None], Qnorm, 0.0)
    Knorm = tl.where(tl.arange(0, BLOCK_SIZE)[None, :] == tl.arange(0, BLOCK_SIZE)[:, None], Knorm, 0.0)

    # compute cosine similarity using dot products
    norm = tl.sqrt(tl.dot(Qnorm, Knorm))
    cos_sim = tl.fdiv(QKt, norm)

    # get diagonal elements of cosine similarity accumulator
    diag_indices = tl.arange(0, BLOCK_SIZE)
    cos_sim = cos_sim[diag_indices, diag_indices]

    # compute probabilities
    p_boundary = tl.clamp((1 - cos_sim) / 2, 0.0, 1.0)
    p_noboundary = 1 - p_boundary

    # compute boundaries
    b = tl.where(p_boundary >= 0.5, True, False)

    # combine probabilities
    p = tl.join(p_noboundary, p_boundary)

    # compute selected probabilities
    sp = tl.maximum(p_noboundary, p_boundary)

    # get offsets for p_noboundary and p_boundary
    p_offsets = tl.arange(0, 2)[None, :]

    # write one row ahead since we place first token manually
    row_offsets = row_offsets + 1

    # compute store memory addresses
    p_ptrs = p_batch_ptr + row_offsets[:, None] * p_row_stride + p_offsets
    b_ptrs = b_batch_ptr + row_offsets[:, None] * b_row_stride
    sp_ptrs = sp_batch_ptr + row_offsets[:, None] * sp_row_stride

    # mask so we don't exceed batch_size * seq_len memory addresses for b, and sp or batch_size * seq_len 
    row_mask = row_offsets < seq_len
    p_mask = row_mask[:, None] & tl.arange(0, 2)[None, :] < 2

    # store results
    tl.store(p_ptrs, p, mask=p_mask)
    tl.store(b_ptrs, b, mask=row_mask)
    tl.store(sp_ptrs, sp, mask=row_mask)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def fused_dc(Q, K):
    assert Q.shape == K.shape, "Q and K must have the same shape"
    assert Q.dim() == 3, "Q and K must be 3D tensors"
    
    batch_size, seq_len, head_dim = Q.shape

    BLOCK_SIZE = triton.next_power_of_2(head_dim)

    num_warps = 8
    num_stages = 4
    BLOCK_SIZE = 64 # 64 to enable WGMMA


    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    p = torch.empty((batch_size, seq_len + 1, 2), device=DEVICE, dtype=torch.bfloat16)
    b = torch.empty((batch_size, seq_len + 1), device=DEVICE, dtype=torch.bool)
    sp = torch.empty((batch_size, seq_len + 1, 1), device=DEVICE, dtype=torch.bfloat16)

    # Create a number of persistent programs.
    fused_dc_kernel_tiled[(batch_size, num_blocks, 1)](
        Q, K, p, b, sp,
        Q.stride(0), K.stride(0), p.stride(0), b.stride(0), sp.stride(0),
        Q.stride(1), K.stride(1), p.stride(1), b.stride(1), sp.stride(1),
        batch_size, seq_len, head_dim,
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
