from typing import List, Union

import torch
import yaml
from torchvision.models import efficientnet_b0

from pytorch_benchmark import benchmark


class CustomImage:
    def __init__(self, data: torch.Tensor):
        self.data = data


class CustomClass:
    def __init__(self, inner_model: torch.nn.Module):
        self.model = inner_model

    def infer(self, custom_image: CustomImage) -> List[torch.Tensor]:
        return [self.model(custom_image.data)]  # wrap in List just because 🤷‍♂️


def test_custom_class():
    # Define custom class
    inner_model = efficientnet_b0()

    if torch.cuda.is_available():
        inner_model = inner_model.cuda()

    custom_class = CustomClass(inner_model)

    # Define custom data
    batch_size = 2
    custom_images = CustomImage(torch.randn(batch_size, 3, 224, 224))  # (B, C, H, W)
    custom_image = CustomImage(torch.randn(1, 3, 224, 224))  # (B, C, H, W)

    # Create helper functions
    def get_device_fn(*args):
        nonlocal custom_class
        return next(custom_class.model.parameters()).device

    def transfer_to_device_fn(
        sample: Union[CustomImage, List[torch.Tensor]],
        device: torch.device,
    ):
        # Input type to CustomClass.infer
        if isinstance(sample, CustomImage):
            return CustomImage(sample.data.to(device=device))

        # Return type from CustomClass.infer
        assert isinstance(sample[0], torch.Tensor)
        return [s.to(device=device) for s in sample]

    results = benchmark(
        model=custom_class.infer,
        sample=custom_images,
        sample_with_batch_size1=custom_image,
        num_runs=10,
        get_device_fn=get_device_fn,
        transfer_to_device_fn=transfer_to_device_fn,
        batch_size=batch_size,
        print_details=True,
    )

    for prop in {"device", "timing"}:  # "flops" and "params" won't work
        assert prop in results

    print(yaml.dump(results))


if __name__ == "__main__":
    test_custom_class()
