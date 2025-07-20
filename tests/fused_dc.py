from hnet.modules.dc import RoutingModule

import torch
import torch.nn.functional as F

if __name__ == "__main__":
    routing_module_unfused = RoutingModule(d_model=1024, device=torch.device("cuda"))
    routing_module_fused = RoutingModule(d_model=1024, device=torch.device("cuda"), fused_dc=True)

    # tie the weights of the two routing modules to ensure identical behavior
    routing_module_unfused.q_proj_layer.weight = routing_module_fused.q_proj_layer.weight
    routing_module_unfused.k_proj_layer.weight = routing_module_fused.k_proj_layer.weight

    x = torch.randn(32, 8192, 1024, device=torch.device("cuda"))
    mask = torch.ones(32, 8192, device=torch.device("cuda"), dtype=torch.bool)

    fused_output = routing_module_fused(x, mask=mask)
    boundary_prob, boundary_mask, selected_probs = fused_output.boundary_prob, fused_output.boundary_mask, fused_output.selected_probs

    unfused_output = routing_module_unfused(x, mask=mask)
    unfused_boundary_prob, unfused_boundary_mask, unfused_selected_probs = unfused_output.boundary_prob, unfused_output.boundary_mask, unfused_output.selected_probs

    # print dtypes
    print(f"boundary_prob dtype: {boundary_prob.dtype}")
    print(f"boundary_mask dtype: {boundary_mask.dtype}")
    print(f"selected_probs dtype: {selected_probs.dtype}")
    print(f"unfused_boundary_prob dtype: {unfused_boundary_prob.dtype}")
    print(f"unfused_boundary_mask dtype: {unfused_boundary_mask.dtype}")
    print(f"unfused_selected_probs dtype: {unfused_selected_probs.dtype}")

    # downcast relevant unfused outputs to float32
    unfused_boundary_prob = unfused_boundary_prob.bfloat16()
    unfused_selected_probs = unfused_selected_probs.bfloat16()

    # # get misses
    miss_mask = (selected_probs != unfused_selected_probs)
    miss_indices = torch.where(miss_mask)
    #print(miss_indices)

    print(f"Number of misses: {miss_indices[-1].shape[0]}")

    print(f"\n max difference: {torch.max(torch.abs(selected_probs - unfused_selected_probs))}")
    relative_errors = torch.abs((selected_probs - unfused_selected_probs) / unfused_selected_probs)  # Already computed, but store it
    max_rel_error = torch.max(relative_errors)  # Find the maximum
    print(f"Max relative error: {max_rel_error.item():.6f} ({max_rel_error.item() * 100:.2f}%)")

    # Check threshold
    threshold = 0.05  # 5%
    if max_rel_error > threshold:
        print(f"WARNING: Max relative error exceeds {threshold*100}% - possible logical mismatch!")
    else:
        print(f"OK: All relative errors <= {threshold*100}% - likely just FP precision noise.")

    assert torch.allclose(boundary_prob, unfused_boundary_prob, atol=8e-3), "fused and unfused boundary_prob are not close enough" # extra check for numerical stability
    assert torch.allclose(boundary_mask, unfused_boundary_mask, atol=8e-3), "fused and unfused boundary_mask are not close enough" # extra check for numerical stability 
    assert torch.allclose(selected_probs, unfused_selected_probs, atol=8e-3), "fused and unfused selected_probs are not close enough" # extra check for numerical stability


