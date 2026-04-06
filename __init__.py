"""
Jupyter Utilities for Virtual Environments and System Monitoring

A collection of utilities for working with Jupyter notebooks, including:
- UV virtual environment management
- Jupyter kernel creation and cleanup
- System monitoring (GPU, CPU, RAM, Disk)
- Helper functions for bash and pip commands
"""

__version__ = "1.0.0"

# Import main utilities
from .helpers import (
    bash,
    pip,
    get_venv_root,
    create_uv_kernel,
    remove_kernel,
    cleanup_session,
)

from .gpu_monitor import (
    SystemMonitor,
    monitor,
)

__all__ = [
    # Helper functions
    "bash",
    "pip",
    "get_venv_root",
    "create_uv_kernel",
    "remove_kernel",
    "cleanup_session",
    # System monitoring
    "SystemMonitor",
    "monitor",
]
