import torch


def is_in_fake_tensor_mode() -> bool:
    # Get the current dispatch mode
    # Check if it's an instance of FakeTensorMode
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None
