# ⏱ pytorch-benchmark
__Easily benchmark model inference FLOPs, latency, throughput, max allocated memory and energy consumption__

<div align="left">
  <a href="https://pypi.org/project/pytorch-benchmark/">
    <img src="https://img.shields.io/pypi/pyversions/pytorch-benchmark" height="20" >
  </a>
  <a href="https://badge.fury.io/py/pytorch-benchmark">
    <img src="https://badge.fury.io/py/pytorch-benchmark.svg" height="20" >
  </a>
  <!-- <a href="https://pepy.tech/project/pytorch-benchmark">
    <img src="https://pepy.tech/badge/pytorch-benchmark/month" height="20">
  </a> -->
  <a href="https://codecov.io/gh/LukasHedegaard/pytorch-benchmark">
    <img src="https://codecov.io/gh/LukasHedegaard/pytorch-benchmark/branch/main/graph/badge.svg?token=B91XGSKSFJ"/>
  </a>
  <a href="https://www.codefactor.io/repository/github/lukashedegaard/pytorch-benchmark/overview/main">
    <img src="https://www.codefactor.io/repository/github/lukashedegaard/pytorch-benchmark/badge/main" alt="CodeFactor" />
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" height="20">
  </a>
</div>

## Install 
```bash
pip install pytorch-benchmark
```

## Usage 
```python
import torch
from torchvision.models import efficientnet_b0
from pytorch_benchmark import benchmark


model = efficientnet_b0()
sample = torch.randn(8, 3, 224, 224)  # (B, C, H, W)
results = benchmark(model, sample, num_runs=100)
```

### Sample results 💻
<details>
  <summary>Macbook Pro (16-inch, 2019), 2.6 GHz 6-Core Intel Core i7</summary>
  
  ```
  device: cpu
  flops: 401669732
  machine_info:
    cpu:
      architecture: x86_64
      cores:
        physical: 6
        total: 12
      frequency: 2.60 GHz
      model: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
    gpus: null
    memory:
      available: 5.86 GB
      total: 16.00 GB
      used: 7.29 GB
    system:
      node: d40049
      release: 21.2.0
      system: Darwin
  params: 5288548
  timing:
    batch_size_1:
      on_device_inference:
        human_readable:
          batch_latency: 74.439 ms +/- 6.459 ms [64.604 ms, 96.681 ms]
          batches_per_second: 13.53 +/- 1.09 [10.34, 15.48]
        metrics:
          batches_per_second_max: 15.478907181264278
          batches_per_second_mean: 13.528026359855625
          batches_per_second_min: 10.343281300091244
          batches_per_second_std: 1.0922382209314958
          seconds_per_batch_max: 0.09668111801147461
          seconds_per_batch_mean: 0.07443853378295899
          seconds_per_batch_min: 0.06460404396057129
          seconds_per_batch_std: 0.006458734193132054
    batch_size_8:
      on_device_inference:
        human_readable:
          batch_latency: 509.410 ms +/- 30.031 ms [405.296 ms, 621.773 ms]
          batches_per_second: 1.97 +/- 0.11 [1.61, 2.47]
        metrics:
          batches_per_second_max: 2.4673319862230025
          batches_per_second_mean: 1.9696935126370148
          batches_per_second_min: 1.6083039834656554
          batches_per_second_std: 0.11341204895590185
          seconds_per_batch_max: 0.6217730045318604
          seconds_per_batch_mean: 0.509410228729248
          seconds_per_batch_min: 0.40529608726501465
          seconds_per_batch_std: 0.030031445467788704
  ```
</details>


## Limitations
Usage assumptions:
- The model has as a `__call__` method that takes the sample, i.e. `model(sample)`.
- The Model also works if the sample had a batch size of 1 (first dimension).

Feature limitataions:
- Allocated memory uses [torch.cuda.max_memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html), which is only available if the model resides on a CUDA device.
- Energy consumption can only be measured on ntel CPU with RAPL support, a NVIDIA GPU.


## Citation
If you like the tool and use it in you research, please consider citing it:
```bibtex
@article{hedegaard2022torchbenchmark,
  title={PyTorch Benchmark},
  author={Lukas Hedegaard},
  journal={GitHub. Note: https://github.com/LukasHedegaard/pytorch-benchmark},
  year={2022}
}
```