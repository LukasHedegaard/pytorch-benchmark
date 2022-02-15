from logging import getLogger
from time import time
from typing import Any, Callable

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
    num_params = _INVALID

    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError as e:
        logger.error(f"Unable to measure model params due to error: {e}")

    return num_params


def measure_allocated_memory(
    model,
    sample,
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    print_details=False,
):
    assert model_device.type == "cuda"

    torch.cuda.reset_peak_memory_stats(device=model_device)
    pre_mem = torch.cuda.memory_allocated(device=model_device)

    transfer_to_device_fn(
        model(transfer_to_device_fn(sample, model_device)),
        "cpu",
    )

    if print_details:
        logger.info(torch.cuda.memory_summary(device=model_device, abbreviated=True))

    post_mem = torch.cuda.memory_allocated(device=model_device)
    max_mem = torch.cuda.max_memory_allocated(device=model_device)

    return pre_mem, post_mem, max_mem


def warm_up(
    model,
    sample,
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    num_runs=10,
    batch_size: int = None,
):
    for _ in tqdm(range(num_runs), desc=f"Warming up with batch_size={batch_size}"):
        transfer_to_device_fn(
            model(transfer_to_device_fn(sample, model_device)),
            "cpu",
        )


def measure_detailed_inference_timing(
    model, sample, model_device, transfer_to_device_fn=torch.Tensor.to
):

    try:
        with torch.autograd.profiler.profile(
            use_cuda=(model_device.type == "cuda"), profile_memory=True
        ) as prof:
            transfer_to_device_fn(
                model(transfer_to_device_fn(sample, model_device)),
                "cpu",
            )

        detailed_timing = prof.key_averages().table(sort_by="self_cpu_time_total")
        logger.info(detailed_timing)

    except Exception as e:
        logger.error(
            f"Caught exception while attempting to measure detailed model inference: {e}"
        )


def measure_repeated_inference_timing(
    model,
    sample,
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    num_runs=100,
    batch_size: int = None,
):

    t_c2d = []
    t_inf = []
    t_d2c = []
    t_tot = []

    for _ in tqdm(
        range(num_runs), desc=f"Measuring inference for batch_size={batch_size}"
    ):
        start_on_cpu = time()
        device_sample = transfer_to_device_fn(sample, model_device)
        start_on_device = time()
        device_result = model(device_sample)
        stop_on_device = time()
        transfer_to_device_fn(device_result, "cpu")
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


def measure_energy(
    model,
    sample,
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    num_runs=100,
    batch_size: int = None,
    include_transfer_costs=True,
    print_fn=logger.info,
):
    inference_joules = _INVALID

    def test_with_transfer():
        nonlocal model, sample
        transfer_to_device_fn(
            model(transfer_to_device_fn(sample, model_device)),
            "cpu",
        )

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

    # except Exception:
    #     pass

    # Try jetson power
    try:
        from .jetson_power import PowerEstimator

        p_est = PowerEstimator(print_fn=print_fn)
        # index 0 is total energy, index 1 is energy over idle consumption:
        meas = []
        for _ in tqdm(
            range(num_runs), desc=f"Measuring energy for batch_size={batch_size}"
        ):
            meas.append(p_est.estimate_fn_power(test_fn)[0] / 1000)
        inference_joules = float(np.array(meas).mean())
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
    get_device_fn: Callable[[Any], torch.device] = get_device,
    transfer_to_device_fn=torch.Tensor.to,
    sample_with_batch_size1: Any = None,
    batch_size: int = None,
    print_fn=logger.info,
):
    results = {}
    batch_size = batch_size or sample.shape[0]

    sample = transfer_to_device_fn(sample, "cpu")

    # Prepare sample with batch size 1
    if sample_with_batch_size1:
        sample1 = sample_with_batch_size1
    else:
        sample1_shape = (1, *sample.shape[1:])
        sample1 = torch.randn(sample1_shape)

    prevously_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    # Get machine info
    machine_info = get_machine_info()
    results["machine_info"] = machine_info
    print_fn(fmt({"Machine info": machine_info}))

    model_device = get_device_fn(model)
    assert isinstance(
        model_device, torch.device
    ), "model_device should be a `torch.device`"
    results["device"] = model_device.type
    print_fn(f"Model device: {model_device}")

    # Measure params
    params = measure_params(model)
    if _is_valid(params):
        results["params"] = params
        print_fn(f"Model parameters: {params} ({format_num(params)})")

    # Measure FLOPs
    try_custom_warmup(model, sample1)

    flops = measure_flops(model, sample1, print_details)
    if _is_valid(flops):
        results["flops"] = flops
        print_fn(f"Model FLOPs: {flops} ({format_num(flops)})")

    # Measure inference timing
    memory = {}
    timing = {}
    energy = {}
    with torch.no_grad():
        for bs in sorted(set([1, batch_size])):
            s = sample1 if bs == 1 else sample

            # Measure Allocated Memory
            if model_device.type == "cuda":
                pre_mem, post_mem, max_mem = measure_allocated_memory(
                    model, sample, model_device, transfer_to_device_fn, print_details
                )
                memory[f"batch_size_{bs}"] = {
                    "pre_inference_bytes": pre_mem,
                    "max_inference_bytes": max_mem,
                    "post_inference_bytes": post_mem,
                    "pre_inference": format_num(pre_mem, bytes=True),
                    "max_inference": format_num(max_mem, bytes=True),
                    "post_inference": format_num(post_mem, bytes=True),
                }
                print_fn(
                    fmt(
                        {
                            f"Memory results (batch_size={bs})": memory[
                                f"batch_size_{bs}"
                            ]
                        }
                    )
                )
            else:
                logger.warning(
                    "Measurement of allocated memory is only available on CUDA devices"
                )

            # Inference timing
            warm_up(
                model,
                s,
                model_device,
                transfer_to_device_fn,
                num_runs=max(1, num_runs // 10),
                batch_size=batch_size,
            )
            if print_details:
                measure_detailed_inference_timing(model, s, model_device)

            timing[f"batch_size_{bs}"] = measure_repeated_inference_timing(
                model,
                s,
                model_device,
                transfer_to_device_fn,
                num_runs,
                batch_size,
            )
            print_fn(
                fmt({f"Timing results (batch_size={bs})": timing[f"batch_size_{bs}"]})
            )

            # Energy measurement
            energy_joules = measure_energy(
                model,
                s,
                model_device,
                transfer_to_device_fn,
                num_runs=max(1, num_runs // 10),
                batch_size=bs,
                include_transfer_costs=True,
                print_fn=print_fn,
            )
            if _is_valid(energy_joules):
                energy_kwh = energy_joules / 3.6e6
                energy[f"batch_size_{bs}"] = {
                    "joules": energy_joules,
                    "kWh": energy_kwh,
                }
                print_fn(
                    fmt(
                        {
                            f"Energy results (batch_size={bs})": energy[
                                f"batch_size_{bs}"
                            ]
                        }
                    )
                )

    results["timing"] = timing
    if energy:
        results["energy"] = energy

    if prevously_training:
        model.train()

    return results
