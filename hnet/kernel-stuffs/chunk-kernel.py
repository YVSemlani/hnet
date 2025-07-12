# hnet/modules/triton_chunking.py
import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def fused_boundary_chunk_kernel(
    # Input tensors
    hidden_states_ptr,      # (B, L, D)
    last_hidden_ptr,        # (B, D) - for boundary prediction
    
    # Output tensors
    chunked_states_ptr,     # (B, max_chunks, D)
    boundary_mask_ptr,      # (B, L)
    boundary_prob_ptr,      # (B, L)
    chunk_counts_ptr,       # (B,) - number of chunks per batch
    
    # Projection weights
    q_weight_ptr,           # (D, D)
    k_weight_ptr,           # (D, D)
    
    # Shapes and strides
    batch_size, seq_len, d_model, max_chunks,
    hs_batch_stride, hs_seq_stride, hs_dim_stride,
    lh_batch_stride, lh_dim_stride,
    cs_batch_stride, cs_chunk_stride, cs_dim_stride,
    
    # Block sizes
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. Boundary prediction (cosine similarity)
    2. Token chunking (gathering boundary tokens)
    """
    
    # Get batch and sequence indices
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)
    
    # Calculate sequence range for this block
    seq_start = seq_block_idx * BLOCK_SIZE_L
    seq_end = tl.minimum(seq_start + BLOCK_SIZE_L, seq_len)
    seq_mask = seq_start + tl.arange(0, BLOCK_SIZE_L) < seq_len
    
    # Load last hidden state for boundary prediction
    last_hidden_offsets = batch_idx * lh_batch_stride + tl.arange(0, BLOCK_SIZE_D)
    last_hidden_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
    last_hidden = tl.load(last_hidden_ptr + last_hidden_offsets, mask=last_hidden_mask)
    
    # Project last hidden state (Q projection)
    q_offsets = tl.arange(0, BLOCK_SIZE_D)[:, None] * d_model + tl.arange(0, BLOCK_SIZE_D)[None, :]
    q_mask = (tl.arange(0, BLOCK_SIZE_D)[:, None] < d_model) & (tl.arange(0, BLOCK_SIZE_D)[None, :] < d_model)
    q_weight = tl.load(q_weight_ptr + q_offsets, mask=q_mask)
    q_proj = tl.dot(last_hidden, q_weight)
    
    # Normalize Q projection
    q_norm = tl.sqrt(tl.sum(q_proj * q_proj))
    q_proj = q_proj / (q_norm + 1e-8)
    
    chunk_count = 0
    
    # Process sequence positions in this block
    for seq_offset in range(seq_start, seq_end):
        if seq_offset >= seq_len:
            break
            
        # Load current hidden state
        hs_offsets = (batch_idx * hs_batch_stride + 
                     seq_offset * hs_seq_stride + 
                     tl.arange(0, BLOCK_SIZE_D))
        hs_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
        current_hidden = tl.load(hidden_states_ptr + hs_offsets, mask=hs_mask)
        
        # Project current hidden state (K projection)
        k_weight = tl.load(k_weight_ptr + q_offsets, mask=q_mask)
        k_proj = tl.dot(current_hidden, k_weight)
        
        # Normalize K projection
        k_norm = tl.sqrt(tl.sum(k_proj * k_proj))
        k_proj = k_proj / (k_norm + 1e-8)
        
        # Compute cosine similarity
        cos_sim = tl.sum(q_proj * k_proj)
        
        # Compute boundary probability
        boundary_prob = tl.maximum(0.0, tl.minimum(1.0, (1.0 - cos_sim) / 2.0))
        
        # Store boundary probability
        prob_offset = batch_idx * seq_len + seq_offset
        tl.store(boundary_prob_ptr + prob_offset, boundary_prob)
        
        # Determine if this is a boundary (threshold = 0.5)
        is_boundary = boundary_prob > 0.5
        
        # Store boundary mask
        tl.store(boundary_mask_ptr + prob_offset, is_boundary.to(tl.int8))
        
        # If boundary, store in chunked output
        if is_boundary and chunk_count < max_chunks:
            chunk_offsets = (batch_idx * cs_batch_stride + 
                           chunk_count * cs_chunk_stride + 
                           tl.arange(0, BLOCK_SIZE_D))
            chunk_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
            tl.store(chunked_states_ptr + chunk_offsets, current_hidden, mask=chunk_mask)
            chunk_count += 1
    
    # Store chunk count for this batch
    if seq_block_idx == 0:  # Only first block writes the count
        tl.store(chunk_counts_ptr + batch_idx, chunk_count)


@triton.jit
def fused_dechunk_ema_kernel(
    # Input tensors
    chunked_states_ptr,     # (B, max_chunks, D)
    boundary_mask_ptr,      # (B, L)
    boundary_prob_ptr,      # (B, L)
    last_ema_ptr,          # (B, D) - EMA state
    
    # Output tensors
    output_states_ptr,      # (B, L, D)
    
    # Shapes and strides
    batch_size, seq_len, d_model, max_chunks,
    cs_batch_stride, cs_chunk_stride, cs_dim_stride,
    os_batch_stride, os_seq_stride, os_dim_stride,
    ema_batch_stride, ema_dim_stride,
    
    # Block sizes
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """
    Fused kernel that combines:
    1. EMA reconstruction from chunked tokens
    2. Scattering back to original sequence length
    """
    
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)
    
    # Calculate sequence range
    seq_start = seq_block_idx * BLOCK_SIZE_L
    seq_end = tl.minimum(seq_start + BLOCK_SIZE_L, seq_len)
    
    # Load EMA state
    ema_offsets = batch_idx * ema_batch_stride + tl.arange(0, BLOCK_SIZE_D)
    ema_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
    ema_state = tl.load(last_ema_ptr + ema_offsets, mask=ema_mask)
    
    chunk_idx = 0
    
    # Process sequence positions
    for seq_offset in range(seq_start, seq_end):
        if seq_offset >= seq_len:
            break
            
        # Load boundary info
        boundary_offset = batch_idx * seq_len + seq_offset
        is_boundary = tl.load(boundary_mask_ptr + boundary_offset)
        boundary_prob = tl.load(boundary_prob_ptr + boundary_offset)
        
        # Clamp boundary probability
        p = tl.maximum(1e-4, tl.minimum(1.0 - 1e-4, boundary_prob))
        
        if is_boundary:
            # Load chunked hidden state
            chunk_offsets = (batch_idx * cs_batch_stride + 
                           chunk_idx * cs_chunk_stride + 
                           tl.arange(0, BLOCK_SIZE_D))
            chunk_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
            chunk_state = tl.load(chunked_states_ptr + chunk_offsets, mask=chunk_mask)
            
            # EMA update: new_value = p * chunk_state + (1 - p) * ema_state
            new_value = p * chunk_state + (1.0 - p) * ema_state
            ema_state = new_value
            chunk_idx += 1
        else:
            # Use EMA state with low boundary probability
            new_value = p * 0.0 + (1.0 - p) * ema_state
            ema_state = new_value
        
        # Store output
        output_offsets = (batch_idx * os_batch_stride + 
                         seq_offset * os_seq_stride + 
                         tl.arange(0, BLOCK_SIZE_D))
        output_mask = tl.arange(0, BLOCK_SIZE_D) < d_model
        tl.store(output_states_ptr + output_offsets, new_value, mask=output_mask)
    
    # Update EMA state
    if seq_block_idx == (seq_len + BLOCK_SIZE_L - 1) // BLOCK_SIZE_L -