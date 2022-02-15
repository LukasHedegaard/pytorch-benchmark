# ⏱ pytorch-benchmark
__Easily benchmark model inference FLOPs, latency, throughput, max allocated memory and energy consumption__

<div align="left">
  <a href="https://pypi.org/project/pytorch-benchmark/">
    <img src="https://img.shields.io/pypi/pyversions/pytorch-benchmark" height="20" >
  </a>
  <a href="https://badge.fury.io/py/pytorch-benchmark">
    <img src="https://badge.fury.io/py/pytorch-benchmark.svg" height="20" >
  </a>
  <a href="https://pepy.tech/project/pytorch-benchmark">
    <img src="https://pepy.tech/badge/pytorch-benchmark/week" height="20">
  </a>
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

<details>
  <summary>Server with NVIDIA GeForce RTX 2080 and Intel Xeon 2.10GHz CPU</summary>
  
  ```
  device: cuda
  flops: 401669732
  machine_info:
    cpu:
      architecture: x86_64
      cores:
        physical: 16
        total: 32
      frequency: 3.00 GHz
      model: Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
    gpus:
    - memory: 8192.0 MB
      name: NVIDIA GeForce RTX 2080
    - memory: 8192.0 MB
      name: NVIDIA GeForce RTX 2080
    - memory: 8192.0 MB
      name: NVIDIA GeForce RTX 2080
    - memory: 8192.0 MB
      name: NVIDIA GeForce RTX 2080
    memory:
      available: 119.98 GB
      total: 125.78 GB
      used: 4.78 GB
    system:
      node: monster
      release: 4.15.0-167-generic
      system: Linux
  max_inference_memory: 736250368
  params: 5288548
  post_inference_memory: 21402112
  pre_inference_memory: 21402112
  timing:
    batch_size_1:
      cpu_to_gpu:
        human_readable:
          batch_latency: "144.815 \xB5s +/- 16.103 \xB5s [136.614 \xB5s, 272.751 \xB5\
            s]"
          batches_per_second: 6.96 K +/- 535.06 [3.67 K, 7.32 K]
        metrics:
          batches_per_second_max: 7319.902268760908
          batches_per_second_mean: 6962.865857677197
          batches_per_second_min: 3666.3496503496503
          batches_per_second_std: 535.0581873859935
          seconds_per_batch_max: 0.0002727508544921875
          seconds_per_batch_mean: 0.00014481544494628906
          seconds_per_batch_min: 0.0001366138458251953
          seconds_per_batch_std: 1.6102982159292097e-05
      gpu_to_cpu:
        human_readable:
          batch_latency: "106.168 \xB5s +/- 17.829 \xB5s [53.167 \xB5s, 248.909 \xB5\
            s]"
          batches_per_second: 9.64 K +/- 1.60 K [4.02 K, 18.81 K]
        metrics:
          batches_per_second_max: 18808.538116591928
          batches_per_second_mean: 9639.942102368092
          batches_per_second_min: 4017.532567049808
          batches_per_second_std: 1595.7983033708472
          seconds_per_batch_max: 0.00024890899658203125
          seconds_per_batch_mean: 0.00010616779327392578
          seconds_per_batch_min: 5.316734313964844e-05
          seconds_per_batch_std: 1.7829135190772566e-05
      on_device_inference:
        human_readable:
          batch_latency: "15.567 ms +/- 546.154 \xB5s [15.311 ms, 19.261 ms]"
          batches_per_second: 64.31 +/- 1.96 [51.92, 65.31]
        metrics:
          batches_per_second_max: 65.31149174711928
          batches_per_second_mean: 64.30692850265713
          batches_per_second_min: 51.918698784442846
          batches_per_second_std: 1.9599322351815833
          seconds_per_batch_max: 0.019260883331298828
          seconds_per_batch_mean: 0.015567030906677246
          seconds_per_batch_min: 0.015311241149902344
          seconds_per_batch_std: 0.0005461537255227954
      total:
        human_readable:
          batch_latency: "15.818 ms +/- 549.873 \xB5s [15.561 ms, 19.461 ms]"
          batches_per_second: 63.29 +/- 1.92 [51.38, 64.26]
        metrics:
          batches_per_second_max: 64.26476266356143
          batches_per_second_mean: 63.28565696640637
          batches_per_second_min: 51.38378232692614
          batches_per_second_std: 1.9198343850767468
          seconds_per_batch_max: 0.019461393356323242
          seconds_per_batch_mean: 0.01581801414489746
          seconds_per_batch_min: 0.015560626983642578
          seconds_per_batch_std: 0.0005498731526138171
    batch_size_8:
      cpu_to_gpu:
        human_readable:
          batch_latency: "805.674 \xB5s +/- 157.254 \xB5s [773.191 \xB5s, 2.303 ms]"
          batches_per_second: 1.26 K +/- 97.51 [434.24, 1.29 K]
        metrics:
          batches_per_second_max: 1293.3407338883749
          batches_per_second_mean: 1259.5653105357776
          batches_per_second_min: 434.23791282741485
          batches_per_second_std: 97.51424036939879
          seconds_per_batch_max: 0.002302885055541992
          seconds_per_batch_mean: 0.000805673599243164
          seconds_per_batch_min: 0.0007731914520263672
          seconds_per_batch_std: 0.0001572538140613121
      gpu_to_cpu:
        human_readable:
          batch_latency: "104.215 \xB5s +/- 12.658 \xB5s [59.605 \xB5s, 128.031 \xB5\
            s]"
          batches_per_second: 9.81 K +/- 1.76 K [7.81 K, 16.78 K]
        metrics:
          batches_per_second_max: 16777.216
          batches_per_second_mean: 9806.840626578907
          batches_per_second_min: 7810.621973929236
          batches_per_second_std: 1761.6008872740726
          seconds_per_batch_max: 0.00012803077697753906
          seconds_per_batch_mean: 0.00010421514511108399
          seconds_per_batch_min: 5.9604644775390625e-05
          seconds_per_batch_std: 1.2658293070174213e-05
      on_device_inference:
        human_readable:
          batch_latency: "16.623 ms +/- 759.017 \xB5s [16.301 ms, 22.584 ms]"
          batches_per_second: 60.26 +/- 2.22 [44.28, 61.35]
        metrics:
          batches_per_second_max: 61.346243290283894
          batches_per_second_mean: 60.25881046175457
          batches_per_second_min: 44.27827629162004
          batches_per_second_std: 2.2193085956672296
          seconds_per_batch_max: 0.02258443832397461
          seconds_per_batch_mean: 0.01662288188934326
          seconds_per_batch_min: 0.01630091667175293
          seconds_per_batch_std: 0.0007590167680596548
      total:
        human_readable:
          batch_latency: "17.533 ms +/- 836.015 \xB5s [17.193 ms, 23.896 ms]"
          batches_per_second: 57.14 +/- 2.20 [41.85, 58.16]
        metrics:
          batches_per_second_max: 58.16374528511205
          batches_per_second_mean: 57.140338855126565
          batches_per_second_min: 41.84762740950632
          batches_per_second_std: 2.1985066663972677
          seconds_per_batch_max: 0.023896217346191406
          seconds_per_batch_mean: 0.01753277063369751
          seconds_per_batch_min: 0.017192840576171875
          seconds_per_batch_std: 0.0008360147274630088
  ```
</details>

... Your turn

## Advanced use
Trying to benchmark a custom class, which is not a `torch.nn.Module`?
- You can pass custom functions to `benchmark` as seen in [this example](tests/test_custom_class.py).

## Limitations
- Allocated memory measurements are only available on CUDA devices.
- Energy consumption can only be measured on NVIDIA Jetson platforms at the moment.
- FLOPs and parameters count is not support for custom classes.


## Citation
If you like the tool and use it in you research, please consider citing it:
```bibtex
@article{hedegaard2022pytorchbenchmark,
  title={PyTorch Benchmark},
  author={Lukas Hedegaard},
  journal={GitHub. Note: https://github.com/LukasHedegaard/pytorch-benchmark},
  year={2022}
}
```
