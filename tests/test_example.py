def test_example():
    import torch
    from torchvision.models import efficientnet_b0

    from torch_benchmark import benchmark

    model = efficientnet_b0()
    sample = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
    results = benchmark(model, sample, num_runs=5)

    for prop in {"device", "flops", "params", "timing"}:
        assert prop in results


if __name__ == "__main__":
    test_example()
