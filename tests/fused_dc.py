from hnet.modules.dc import RoutingModule

import torch
import torch.nn.functional as F

def print_comparison(title, items):
    print(f"\n{title}:")
    for name, fused, unfused in items:
        print(f"{name:<20} | Fused: {str(fused):<15} | Unfused: {str(unfused)}")

if __name__ == "__main__":
    # Initialize modules
    routing_module_unfused = RoutingModule(d_model=1024, device=torch.device("cuda"))
    routing_module_fused = RoutingModule(d_model=1024, device=torch.device("cuda"), fused_dc=True)

    # Tie weights for comparable results
    routing_module_unfused.q_proj_layer.weight = routing_module_fused.q_proj_layer.weight
    routing_module_unfused.k_proj_layer.weight = routing_module_fused.k_proj_layer.weight

    # Generate test data
    x = torch.randn(2, 8192, 1024, device=torch.device("cuda"))
    mask = torch.ones(2, 8192, device=torch.device("cuda"), dtype=torch.bool)

    # Run inference
    fused_output = routing_module_fused(x, mask=mask)
    unfused_output = routing_module_unfused(x, mask=mask)

    # Extract outputs
    (boundary_prob, boundary_mask, selected_probs) = (fused_output.boundary_prob, 
                                                     fused_output.boundary_mask,
                                                     fused_output.selected_probs)
    (unfused_boundary_prob, unfused_boundary_mask, 
     unfused_selected_probs) = (unfused_output.boundary_prob,
                              unfused_output.boundary_mask,
                              unfused_output.selected_probs)

    # Print metadata comparison
    print_comparison("Metadata", [
        ("dtype", boundary_prob.dtype, unfused_boundary_prob.dtype),
        ("shape", str(boundary_prob.shape), str(unfused_boundary_prob.shape))
    ])

    # Convert unfused outputs for comparison
    unfused_boundary_prob = unfused_boundary_prob.to(torch.bfloat16)
    unfused_selected_probs = unfused_selected_probs.to(torch.bfloat16)

    # Validate results
    mismatch_count = 0
    for fused, unfused, name in [
        (boundary_prob, unfused_boundary_prob, "boundary_prob"),
        (boundary_mask, unfused_boundary_mask, "boundary_mask"),
        (selected_probs, unfused_selected_probs, "selected_probs")
    ]:
        miss_mask = (fused != unfused)
        miss_count = miss_mask.sum().item()
        
        print(f"\nChecking {name}:")
        print(f"Mismatches: {miss_count}/{fused.numel()} ({miss_count/fused.numel():.2%})")
        
        if miss_count > 0:
            # Handle boolean tensors differently
            if fused.dtype == torch.bool:
                print(f"Boolean tensor - showing mismatch locations only")
            else:
                abs_diff = torch.abs(fused - unfused)
                print(f"Max absolute difference: {abs_diff.max().item():.4f}")
                print(f"Mean absolute difference: {abs_diff.mean().item():.4f}")
            
            # Show first few mismatches with their indices and values
            miss_indices = torch.nonzero(miss_mask, as_tuple=True)
            print(f"First 10 mismatch locations:")
            for i in range(min(10, miss_count)):
                idx = tuple(miss_indices[j][i] for j in range(len(miss_indices)))
                fused_val = fused[idx].item()
                unfused_val = unfused[idx].item()
                
                if fused.dtype == torch.bool:
                    print(f"  Index {idx}: Fused={fused_val}, Unfused={unfused_val}")
                else:
                    diff = abs(fused_val - unfused_val)
                    print(f"  Index {idx}: Fused={fused_val:.6f}, Unfused={unfused_val:.6f}, Diff={diff:.6f}")
            
            if miss_count > 10:
                print(f"  ... and {miss_count - 10} more mismatches")

    # Final validation
    assert torch.allclose(selected_probs, unfused_selected_probs, atol=8e-3), "Selected probs mismatch"
    assert torch.allclose(boundary_prob, unfused_boundary_prob, atol=8e-3), "Boundary prob mismatch"
    assert torch.allclose(boundary_mask, unfused_boundary_mask, atol=8e-3), "Boundary mask mismatch"
    print("\nAll comparisons within expected tolerances!")
