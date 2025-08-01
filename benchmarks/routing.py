import triton
import triton.language as tl
from triton.runtime import driver

from hnet.modules.dc import RoutingModule
from hnet.modules.ops.fused_dc import fused_dc

import torch
import torch.nn.functional as F
import gc
import pandas as pd

DEVICE = triton.runtime.driver.active.get_active_torch_device()

RUN_NAME = "Routing Module Benchmark"

def ms_to_tb_s(x, ms):
    return x.shape[0] * x.shape[1] * x.shape[2] * 4 * 1000 / ms / 1e12

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BATCH_SIZE'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(0, 9)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Fused",
            "Unfused",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="Throughput (TB/s)",  # label name for the y-axis
        plot_name=RUN_NAME,  # name for the plot. Used also as a file name for saving the plot.
        args={'SEQ_LEN': 8192, 'HEAD_DIM': 1024},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(BATCH_SIZE, SEQ_LEN, HEAD_DIM, provider):
    routing_module_unfused = RoutingModule(d_model=HEAD_DIM, device=torch.device("cuda"))
    routing_module_unfused = torch.compile(routing_module_unfused)
    routing_module_fused = RoutingModule(d_model=HEAD_DIM, device=torch.device("cuda"), fused_dc=True)
    #routing_module_fused = torch.compile(routing_module_fused)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=torch.device("cuda"))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, device=torch.device("cuda"), dtype=torch.bool)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: routing_module_unfused(x, mask=mask), warmup=10)
        #print(f"Unfused DC: {ms}")
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: routing_module_fused(x, mask=mask), warmup=10)
        #print(f"Fused DC: {ms}")

    num_memory_ops = 2
    tbps = lambda ms: num_memory_ops * x.numel() * x.element_size() * 1e-12 / (ms * 1e-3)

    # clear cache
    torch.cuda.empty_cache()
    gc.collect()
    return tbps(ms)

if __name__ == "__main__":
    save_path = "reports/routing"
    benchmark.run(show_plots=True, print_data=True, save_path=save_path)

    # add a percent speedup to the saved csv
    df = pd.read_csv(f"{save_path}/{RUN_NAME}.csv")
    df["% Speedup"] = (df["Fused"] - df["Unfused"]) / df["Unfused"] * 100
    df.to_csv(f"{save_path}/{RUN_NAME}.csv", index=False)
