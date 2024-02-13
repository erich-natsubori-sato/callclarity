import torch
from loguru import logger

def get_torch_device(return_str:bool = True):

    # If there's a GPU available, use it
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU
        if return_str: device = "cuda:0"
        else: device = torch.device("cuda:0")

        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()

        logger.info(f"Using GPU '{device_name}' as the torch device")
        logger.info(f"└ There are {device_count:0.0f} GPU(s) available")

    # If not, try using apple silicon (MPS)
    elif torch.backends.mps.is_available():

        # Tell PyTorch to use the MPS
        if return_str: device = "mps"
        else: device = torch.device("mps")

        logger.info(f"Using Apple Silicon as the torch device")

    # If not, use CPU
    else:
        if return_str: device = "cpu"
        else: device = torch.device("cpu")

        logger.info(f"Using CPU as the torch device")

    return device