import itertools
import os
from typing import Union

import torch


def get_default_device():
    if torch.cuda.is_available():
        return get_most_free_gpu()
    elif torch.backends.mps.is_available():
        # Not all operations implemented in MPS yet
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1"
        if use_mps:
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def get_available_devices(num_devices=None, max_devices=None):
    device = get_default_device()
    if device.type == "cuda":
        num_gpus = torch.cuda.device_count()
        if max_devices is not None:
            num_gpus = min(num_gpus, max_devices)
        gpu_list = [get_device(i) for i in range(num_gpus)]
        if num_devices is not None:
            cycle_gpu_list = itertools.cycle(gpu_list)
            gpu_list = [next(cycle_gpu_list) for _ in range(num_devices)]
        return gpu_list
    else:
        num_devices = num_devices if num_devices else torch.multiprocessing.cpu_count()
        num_devices = min(num_devices, max_devices) if max_devices is not None else num_devices
        return [device]*num_devices


def get_device(gpu_id: int):
    if gpu_id is not None and torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu_id}')
    else:
        return None


def get_device_mem_info(device: Union[int, torch.device]):
    if not torch.cuda.is_available():
        return None

    free_mem, total_mem = torch.cuda.mem_get_info(device=device)
    return free_mem, total_mem


def get_most_free_gpu():
    """
    Get the GPU with the most free memory
    If system has no GPUs (or CUDA not available), return None
    """
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return None

    most_free_idx, most_free_val = -1, -1
    for i in range(num_gpus):
        free_mem, total_mem = get_device_mem_info(i)
        if free_mem > most_free_val:
            most_free_idx, most_free_val = i, free_mem

    return torch.device(f'cuda:{most_free_idx}')
