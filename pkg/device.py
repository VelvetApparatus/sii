import torch

def _device():

    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = torch.device(_device())


def get_device():
    print("DEVICE: ", device)
    return device