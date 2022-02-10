from logging import getLogger
from time import time

import numpy as np
import torch
import yaml
from ptflops import get_model_complexity_info
from tqdm import tqdm

from .format import format_num, format_time
from .machine_info import get_machine_info

logger = getLogger("torch-benchmark")
_INVALID = float("nan")


def _is_valid(val):
    return val == val


def try_custom_warmup(model, shape):
    # Allow custom warm-up function defined in model, e.g.
    # class Model:
    #
    #   def warm_up(self, input_shape: List[int]):
    #       ...
    success = False
    try:
        if hasattr(model, "warm_up"):
            model.warm_up(shape)
            success = True
    except Exception:
        pass

    return success


def measure_flops(model, sample, print_details=False):
    flops = _INVALID
    try:
        flops, _ = get_model_complexity_info(
            model,
            tuple(sample.shape[1:]),
            as_strings=False,
            print_per_layer_stat=print_details,
            verbose=print_details,
        )
        flops = int(flops)
    except Exception as e:  # pragma: no cover
        logger.error(f"Unable to measure model FLOPs due to error: {e}")

    return flops


def get_device(model):
    return next(model.parameters()).device


def measure_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def measure_allocated_memory(model, sample, print_details=False):
    model_device = get_device(model)
    assert model_device.type == "cuda"

    torch.cuda.reset_peak_memory_stats(device=model_device)
    pre_mem = torch.cuda.memory_allocated(device=model_device)

    model(sample.to(device=model_device)).to(device="cpu")

    if print_details:
        logger.info(torch.cuda.memory_summary(device=model_device, abbreviated=True))

    post_mem = torch.cuda.memory_allocated(device=model_device)
    max_mem = torch.cuda.max_memory_allocated(device=model_device)

    return pre_mem, post_mem, max_mem


def warm_up(model, sample, num_runs=10):
    model_device = get_device(model)
    batch_size = sample.shape[0]
    for _ in tqdm(range(num_runs), desc=f"Warming up with batch_size={batch_size}"):
        model(sample.to(device=model_device)).to(device="cpu")


def measure_detailed_inference_timing(model, sample):
    model_device = get_device(model)

    try:
        with torch.no_grad(), torch.autograd.profiler.profile(
            use_cuda=(model_device.type == "cuda"), profile_memory=True
        ) as prof:
            model(sample.to(device=model_device)).to(device="cpu")

        detailed_timing = prof.key_averages().table(sort_by="self_cpu_time_total")
        logger.info(detailed_timing)

    except Exception as e:
        logger.error(
            f"Caught exception while attempting to measure detailed model inference: {e}"
        )


def measure_repeated_inference_timing(model, sample, num_runs):
    model_device = get_device(model)

    batch_size = sample.shape[0]

    t_c2d = []
    t_inf = []
    t_d2c = []
    t_tot = []

    for _ in tqdm(
        range(num_runs), desc=f"Measuring inference with batch_size={batch_size}"
    ):
        start_on_cpu = time()
        device_sample = sample.to(device=model_device)
        start_on_device = time()
        device_result = model(device_sample)
        stop_on_device = time()
        device_result.to(device="cpu")  # discard result
        stop_on_cpu = time()

        t_c2d.append(start_on_device - start_on_cpu)
        t_inf.append(stop_on_device - start_on_device)
        t_d2c.append(stop_on_cpu - stop_on_device)
        t_tot.append(stop_on_cpu - start_on_cpu)

    results_dict = {}

    times_and_titles = [(t_inf, "on_device_inference")]
    if model_device.type == "cuda":
        times_and_titles.extend(
            [
                (t_c2d, "cpu_to_gpu"),
                (t_d2c, "gpu_to_cpu"),
                (t_tot, "total"),
            ]
        )

    for s_per_batch, title in times_and_titles:
        s_per_batch = np.array(s_per_batch)
        batches_per_s = 1 / s_per_batch

        metrics = {
            "batches_per_second_mean": float(batches_per_s.mean()),
            "batches_per_second_std": float(batches_per_s.std()),
            "batches_per_second_min": float(batches_per_s.min()),
            "batches_per_second_max": float(batches_per_s.max()),
            "seconds_per_batch_mean": float(s_per_batch.mean()),
            "seconds_per_batch_std": float(s_per_batch.std()),
            "seconds_per_batch_min": float(s_per_batch.min()),
            "seconds_per_batch_max": float(s_per_batch.max()),
        }

        human_readable = {
            "batches_per_second": f"{format_num(batches_per_s.mean())} +/- {format_num(batches_per_s.std())} [{format_num(batches_per_s.min())}, {format_num(batches_per_s.max())}]",
            "batch_latency": f"{format_time(s_per_batch.mean())} +/- {format_time(s_per_batch.std())} [{format_time(s_per_batch.min())}, {format_time(s_per_batch.max())}]",
        }

        results_dict[title] = {"metrics": metrics, "human_readable": human_readable}

    return results_dict


def measure_energy(model, sample, print_details=False, include_transfer_costs=True):
    inference_joules = _INVALID
    model_device = get_device(model)

    def test_with_transfer():
        nonlocal model, sample
        model(sample.to(device=model_device)).to(device="cpu")

    def test_without_transfer():
        nonlocal model, sample
        model(sample)

    if include_transfer_costs:
        test_fn = test_with_transfer
    else:
        test_fn = test_without_transfer
        sample = sample.to(model_device)

    # # Try carbon-tracker: The library is still too young
    # try:
    #     from carbontracker import tracker

    #     # Check if components are available (TODO: find a less brittle implementation for this)
    #     pids = tracker.CarbonTracker._get_pids(None)
    #     components = tracker.component.create_components(
    #         components="all", pids=pids, devices_by_pid=False
    #     )
    #     if not any([cmp for cmp in components if cmp.available()]):
    #         raise Exception("Valid CarbonTracker device not available")

    #     tracker = tracker.CarbonTracker(epochs=1, verbose=print_details)
    #     tracker.epoch_start()
    #     test_fn()
    #     tracker.epoch_end()

    #     # Grab results from logger
    #     # TODO

    # except Exception:
    #     pass

    # Try jetson power
    try:
        from .jetson_power import PowerEstimator

        p_est = PowerEstimator()
        # index 0 is total energy, index 1 is energy over idle consumption:
        total_joules = p_est.estimate_fn_power(test_fn)[0] / 1000
        inference_joules = total_joules
    except Exception:
        pass

    if not _is_valid(inference_joules):
        logger.error(
            "Unable to measure energy consumption. Device must be a NVIDIA Jetson."
        )

    return inference_joules


def fmt(d: dict):
    return yaml.dump(d)


def benchmark(
    model: torch.nn.Module,
    sample: torch.Tensor,
    num_runs: int = 100,
    print_details=False,
):
    results = {}
    batch_size, rest_shape = sample.shape[0], sample.shape[1:]

    sample = sample.to(device="cpu")

    # Prepare sample with batch size 1
    sample1_shape = (1, *rest_shape)
    sample1 = torch.randn(sample1_shape)

    prevously_training = model.training
    model.eval()

    # Get machine info
    machine_info = get_machine_info()
    results["machine_info"] = machine_info
    logger.info(fmt({"Machine info": machine_info}))

    model_device = get_device(model)
    results["device"] = model_device.type
    logger.info(f"Model device: {model_device}")

    # Measure params
    params = measure_params(model)
    results["params"] = params
    logger.info(f"Model parameters: {params} ({format_num(params)})")

    # Measure FLOPs
    try_custom_warmup(model, sample1)

    flops = measure_flops(model, sample1, print_details)
    if _is_valid(flops):
        results["flops"] = flops
        logger.info(f"Model FLOPs: {flops} ({format_num(flops)})")

    # Measure Allocated Memory
    if model_device.type == "cuda":
        pre_mem, post_mem, max_mem = measure_allocated_memory(
            model, sample, print_details
        )
        results["pre_inference_memory"] = pre_mem
        results["max_inference_memory"] = max_mem
        results["post_inference_memory"] = post_mem
        logger.info(
            f"Allocated GPU memory prior to inference: {pre_mem} ({format_num(pre_mem, bytes=True)})"
        )
        logger.info(
            f"Allocated GPU memory after to inference: {post_mem} ({format_num(post_mem, bytes=True)})"
        )
        logger.info(
            f"Max allocated GPU memory during inference: {max_mem} ({format_num(max_mem, bytes=True)})"
        )
    else:
        logger.warning(
            "Measurement of allocated memory is only available on CUDA devices"
        )

    # Measure inference timing
    timing = {}
    energy = {}
    for bs in set([1, batch_size]):
        s = sample1 if bs == 1 else sample

        # Inference timing
        warm_up(model, s, num_runs=max(5, num_runs // 10))
        if print_details:
            measure_detailed_inference_timing(model, s)

        timing[f"batch_size_{bs}"] = measure_repeated_inference_timing(
            model, s, num_runs
        )
        logger.info(
            fmt({f"Timing results (batch_size={bs})": timing[f"batch_size_{bs}"]})
        )

        # Energy measurement
        energy_joules = measure_energy(model, sample, print_details)
        if _is_valid(energy_joules):
            energy_kwh = energy_joules / 3.6e6
            energy[f"batch_size_{bs}"] = {
                "joules": energy_joules,
                "kWh": energy_kwh,
            }
            logger.info(f"Inference energy: {energy_joules} J ({energy_kwh} kWh)")

    results["timing"] = timing
    if energy:
        results["energy"] = energy

    if prevously_training:
        model.train()

    return results
