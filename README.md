# Jupyter Utils

Personal utilities for Jupyter notebooks. Handles virtual environment management and system monitoring.

## What's included

- UV virtual environment management and Jupyter kernel creation
- pip helper with automatic venv detection (uv or regular pip)
- System monitor widget for GPU, CPU, RAM, and disk
- Bash command execution
- Kernel cleanup utilities

## Installation

Clone to home directory:

```bash
cd ~
git clone https://github.com/h-a-s-k/jupyter-utils.git
```

Add to Python path in notebooks:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'jupyter-utils'))
# or
# sys.path.insert(0, os.path.expanduser('~/jupyter-utils'))

from helpers import create_uv_kernel, cleanup_session, pip, bash, get_venv_root, remove_kernel
from gpu_monitor import SystemMonitor, monitor
from training_callbacks import EarlyStoppingCallback, TimeLimitCallback, TrainingLoggerCallback
```

## Usage

### Create virtual environment

```python
from helpers import create_uv_kernel

create_uv_kernel(
    env_dir="my_env",
    kernel_name="my_env",
    display_name="My Environment",
    python_version="3.12"
)
```

#### Then: Kernel → Change kernel → "My Environment"

### Install packages

```python
from helpers import pip

pip("install torch torchvision")
pip("install transformers")
pip("list")
```

### Run bash commands

```python
from helpers import bash

bash("ls -la")
bash("nvidia-smi")
```

### Monitor system

```python
from gpu_monitor import monitor

mon = monitor()
# ... do work ...
mon.stop()
```

### Cleanup

```python
from helpers import cleanup_session

cleanup_session("my_env", remove_venv=True)
```

## Details

### Virtual environments

```python
from helpers import create_uv_kernel, remove_kernel, cleanup_session

# Create
create_uv_kernel(
    env_dir="ml_env",
    kernel_name="ml_kernel",
    display_name="ML Project",
    python_version="3.11"
)

# Remove kernel only
remove_kernel("ml_kernel")

# Remove kernel and venv
cleanup_session("ml_kernel", remove_venv=True)
```

### Package management

```python
from helpers import pip

pip("install numpy pandas")
pip("install git+https://github.com/user/repo.git")
pip("install torch --index-url https://download.pytorch.org/whl/rocm6.0")
pip("list")

# Silent mode
result = pip("show torch", show_output=False)
```

Uses uv pip if available, falls back to regular pip. Always installs to correct venv.

### Bash

```python
from helpers import bash

bash("echo 'test'")
bash("rocm-smi")

# Silent mode
result = bash("pwd", show_output=False)
print(result.stdout)
```

### Silent operations

```python
from helpers import bash, pip

# Run commands silently
result = bash("ls -la", show_output=False)
if result.returncode == 0:
    print("Success!")
    print(result.stdout)

# Check if package is installed
result = pip("show torch", show_output=False)
if "Version:" in result.stdout:
    print("Torch is installed")
```

### System monitor

```python
from gpu_monitor import monitor

mon = monitor(refresh_interval=2.0)
mon.stop()

# Manual control
mon = monitor(auto_start=False)
mon.start()
mon.stop()
```

Monitors GPU (rocm-smi/nvidia-smi/amd-smi), CPU, RAM, and disk. Runs in background thread.

### Training callbacks

```python
from training_callbacks import EarlyStoppingCallback, TimeLimitCallback, TrainingLoggerCallback

# Early stopping - stops if loss doesn't improve
early_stop = EarlyStoppingCallback(
    patience=50,                           # Steps to wait without improvement
    min_delta=0.01,                        # Minimum change to count as improvement
    metric_name="eval_assistant_accuracy", # Explicitly track a custom loss metric
	greater_is_better=True,                # Accuracy/Loss
    stop_file_dir=".",                     # Directory to check for manual STOP file
    verbose=False                          # Print progress messages
)

# Time limit - stops after specified hours
time_limit = TimeLimitCallback(max_hours=3.0)

# Training logger - saves metrics to text and JSON
logger = TrainingLoggerCallback(
    base_path="./logs",
    extra_config={"max_seq_length": 2048, "lora_r": 16}  # Optional
)

# Use with trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(...),
    callbacks=[early_stop, time_limit, logger]
)
```

EarlyStoppingCallback stops when loss plateaus or you create a STOP file in the working directory. TimeLimitCallback stops after a time limit. TrainingLoggerCallback logs everything to text and JSON files.

## Example

```python
import sys, os
sys.path.insert(0, os.path.expanduser('~/jupyter-utils'))

from helpers import create_uv_kernel, pip, cleanup_session
from gpu_monitor import monitor

create_uv_kernel("ml_env", "ml", "ML Environment")
# Switch kernel
pip("install torch transformers")
mon = monitor()
# ... work ...
mon.stop()
# Remove kernel and venv
cleanup_session("ml", remove_venv=True)
```

## Requirements

- Python 3.8+
- psutil
- ipywidgets

```bash
pip install psutil ipywidgets
```

## Notes

This is a personal utility collection. Not intended for public use or contributions.
