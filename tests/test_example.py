import torch
import yaml
from torchvision.models import efficientnet_b0

from pytorch_benchmark import benchmark


def test_example():
    model = efficientnet_b0()

    if torch.cuda.is_available():
        model = model.cuda()

    sample = torch.randn(2, 3, 224, 224)  # (B, C, H, W)

    results = benchmark(model, sample, num_runs=10, print_details=True)

    for prop in {"device", "flops", "params", "timing"}:
        assert prop in results

    print(yaml.dump(results))


if __name__ == "__main__":
    test_example()
