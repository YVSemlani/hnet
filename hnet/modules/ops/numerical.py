from hnet.modules.ops.fused_dc import fused_dc
from hnet.modules.dc import RoutingModule

import torch
import torch.nn.functional as F


def run_fused(x, routing_module):
    q_proj_layer = routing_module.q_proj_layer
    k_proj_layer = routing_module.k_proj_layer

    Q = q_proj_layer(x[:, :-1])
    K = k_proj_layer(x[:, 1:])

    #print(Q.shape)
    #print(K.shape)

    p, b = fused_dc(Q, K)

    print(p)
    #print(b)

    return p, b

if __name__ == "__main__":
    routing_module = RoutingModule(d_model=1024, device=torch.device("cuda"))

    x = torch.randn(32, 8192, 1024, device=torch.device("cuda"))
    mask = torch.ones(32, 8192, device=torch.device("cuda"), dtype=torch.bool)

    routing_module_output = routing_module(x, mask=mask)
    boundary_prob, boundary_mask, selected_probs = routing_module_output.boundary_prob, routing_module_output.boundary_mask, routing_module_output.selected_probs
    boundary_prob = boundary_prob[:, :, 1]
    
    print(boundary_prob.shape)
    #print(boundary_mask)

    p, b = run_fused(x, routing_module)

    print(p.shape)

    # get misses
    miss_mask = (p != boundary_prob)
    miss_indices = torch.where(miss_mask)
    #print(miss_indices)

    print(f"Number of misses: {miss_indices[-1].shape[0]}")

    print(f"\n max difference: {torch.max(torch.abs(p - boundary_prob))}")
    relative_errors = torch.abs((p - boundary_prob) / boundary_prob)  # Already computed, but store it
    max_rel_error = torch.max(relative_errors)  # Find the maximum
    print(f"Max relative error: {max_rel_error.item():.6f} ({max_rel_error.item() * 100:.2f}%)")

    # Check threshold
    threshold = 0.05  # 5%
    if max_rel_error > threshold:
        print(f"WARNING: Max relative error exceeds {threshold*100}% - possible logical mismatch!")
    else:
        print(f"OK: All relative errors <= {threshold*100}% - likely just FP precision noise.")

    assert torch.allclose(p, boundary_prob, atol=1e-4), "p and boundary_prob are not close enough" # extra check for numerical stability