from platform import uname
from typing import Any, Dict

from cpuinfo import get_cpu_info
from GPUtil import getGPUs
from psutil import cpu_count, cpu_freq, virtual_memory

from .format import format_num


def get_machine_info() -> Dict[str, Any]:
    sys = uname()
    cpu = get_cpu_info()
    svmem = virtual_memory()
    gpus = getGPUs()

    return {
        "system": {"system": sys.system, "node": sys.node, "release": sys.release},
        "cpu": {
            "model": cpu["brand_raw"],
            "architecture": cpu["arch_string_raw"],
            "cores": {
                "physical": cpu_count(logical=False),
                "total": cpu_count(logical=True),
            },
            "frequency": f"{(cpu_freq().max / 1000):.2f} GHz",
        },
        "memory": {
            "total": format_num(svmem.total, bytes=True),
            "used": format_num(svmem.used, bytes=True),
            "available": format_num(svmem.available, bytes=True),
        },
        "gpus": (
            [{"name": g.name, "memory": f"{g.memoryTotal} MB"} for g in gpus]
            if gpus
            else None
        ),
    }
