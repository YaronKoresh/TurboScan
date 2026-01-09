import os
from typing import List
from dataclasses import dataclass, field

@dataclass
class HardwareConfig:
    cpu_count: int = field(default_factory=lambda: os.cpu_count() or 1)
    cpu_count_physical: int = field(default_factory=lambda: os.cpu_count() or 1)
    memory_total: int = 0
    memory_available: int = 0
    gpu_count: int = 0
    gpu_memory: List[int] = field(default_factory=list)
    gpu_names: List[str] = field(default_factory=list)
    numa_nodes: int = 1
    cache_line_size: int = 64
    page_size: int = 4096

def detect_hardware() -> HardwareConfig:
    config = HardwareConfig()
    try:
        config.cpu_count = os.cpu_count() or 1
        try:
            import psutil
            config.cpu_count_physical = psutil.cpu_count(logical=False) or config.cpu_count
            mem = psutil.virtual_memory()
            config.memory_total = mem.total
            config.memory_available = mem.available
        except ImportError:
            config.cpu_count_physical = config.cpu_count
            config.memory_total = 8 * 1024 ** 3
            config.memory_available = 4 * 1024 ** 3
    except:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            config.gpu_count = torch.cuda.device_count()
            for i in range(config.gpu_count):
                props = torch.cuda.get_device_properties(i)
                config.gpu_memory.append(props.total_memory)
                config.gpu_names.append(props.name)
    except ImportError:
        pass
    try:
        config.page_size = os.sysconf('SC_PAGESIZE')
    except:
        config.page_size = 4096
    return config

HARDWARE = detect_hardware()
