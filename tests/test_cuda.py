import torch


def test_cuda_is_available():
    # Test that CUDA is properly installed
    assert torch.cuda.is_available()


def test_cuda_device_count():
    # Test that two GPUs are available
    assert torch.cuda.device_count() == 2
