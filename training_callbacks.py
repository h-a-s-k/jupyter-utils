"""
Training Callbacks for Transformers Trainer

Provides callbacks for:
1. Early stopping based on loss improvement
2. Time-limited training sessions
3. Comprehensive training logging to text and JSON files
"""

import time
import json
import os
from datetime import datetime
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# Shared state for callbacks to communicate stop reasons
_CALLBACK_STOP_REASON = None


def set_callback_stop_reason(reason):
    """Set the global stop reason that will be picked up by the logger."""
    global _CALLBACK_STOP_REASON
    _CALLBACK_STOP_REASON = reason


def get_callback_stop_reason():
    """Get the global stop reason set by any callback."""
    return _CALLBACK_STOP_REASON


def clear_callback_stop_reason():
    """Clear the global stop reason (called at training start)."""
    global _CALLBACK_STOP_REASON
    _CALLBACK_STOP_REASON = None


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback that stops training if metric doesn't improve.
    
    Automatically detects and prioritizes available metrics:
    1. Custom metric (if metric_name provided)
    2. Eval loss (if available): eval_loss
    3. Training loss (fallback): loss
    
    Args:
        patience (int): Number of logging checks to wait for improvement. Default: 10
        min_delta (float): Minimum change in metric to qualify as improvement. Default: 0.0
        metric_name (str): Specific metric to track (e.g., 'eval_assistant_accuracy'). 
                          If None, auto-detects eval metrics. Default: None
        greater_is_better (bool): If True, higher values are better (for accuracy).
                                 If False, lower values are better (for loss).
                                 Only used when metric_name is specified. Default: None (auto-detect)
        verbose (bool): Whether to print progress messages. Default: False
        stop_file_dir (str): Directory to check for manual stop file. Default: "."
        
    Note:
        Patience is measured in logging checks, not training steps. For example, if
        logging_steps=100 and patience=5, training will stop after 5 consecutive
        logging events (500 steps) without improvement.
        
    Example:
        # Auto-detect eval metrics (recommended for loss)
        callback = EarlyStoppingCallback(patience=10, min_delta=0.01)
        
        # Track accuracy metric (higher is better)
        callback = EarlyStoppingCallback(
            patience=10, 
            metric_name='eval_assistant_accuracy',
            greater_is_better=True
        )
        
        # Track loss metric (lower is better)
        callback = EarlyStoppingCallback(
            patience=10,
            metric_name='eval_assistant_loss',
            greater_is_better=False
        )
    """
    def __init__(self, patience=10, min_delta=0.0, metric_name=None, greater_is_better=None, verbose=False, stop_file_dir="."):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.verbose = verbose
        self.stop_file_dir = stop_file_dir
        self.stop_file_path = os.path.join(stop_file_dir, "STOP")
        
        # State tracking
        self.best_metric = None
        self.best_step = 0
        self.wait_count = 0
        self.check_count = 0
        self.stopped_early = False
        self.stopped_manually = False
        self.stop_reason = None
        self.detected_metric_name = None
        self.detected_greater_is_better = None
        self._warned_metrics = set()
    
    def _auto_detect_direction(self, metric_name):
        """Auto-detect whether higher or lower is better based on metric name."""
        metric_lower = metric_name.lower()
        
        # Higher is better
        if any(x in metric_lower for x in ['accuracy', 'acc', 'f1', 'precision', 'recall', 'bleu', 'rouge']):
            return True
        
        # Lower is better
        if 'loss' in metric_lower:
            return False
        
        # Default: lower is better (conservative)
        return False
    
    def _detect_metric_name(self, logs):
        """Auto-detect which metric to use based on what's available in logs."""
        if not logs:
            return None
        
        # Priority 1: Custom eval metrics (eval_* but not eval_loss)
        for key in logs.keys():
            if key.startswith('eval_') and key != 'eval_loss':
                return key
        
        # Priority 2: Standard eval loss
        if 'eval_loss' in logs:
            return 'eval_loss'
        
        # Priority 3: Training loss fallback
        if 'loss' in logs:
            return 'loss'
        
        return None
    
    def _initialize_metric_tracking(self, metric_name, is_eval=False):
        """Initialize metric tracking with proper direction detection."""
        self.detected_metric_name = metric_name
        self.detected_greater_is_better = (
            self.greater_is_better if self.greater_is_better is not None
            else self._auto_detect_direction(metric_name)
        )
        direction = "higher is better" if self.detected_greater_is_better else "lower is better"
        metric_type = "eval" if is_eval else "training"
        print(f"✓ Using specified {metric_type} metric: {metric_name} ({direction})")
    
    def _get_metric_value(self, logs, is_eval=False):
        """Extract metric value from logs, handling initialization and fallbacks.
        
        Returns:
            tuple: (metric_value, should_continue) where should_continue is False if we should skip this check
        """
        # Skip eval metrics in on_log (they come through on_evaluate)
        if not is_eval and self.metric_name and self.metric_name.startswith('eval_'):
            return None, False
        
        # User specified a metric
        if self.metric_name:
            if self.metric_name not in logs:
                # Fallback logic
                if self.detected_metric_name is None:
                    fallback = 'eval_loss' if is_eval and 'eval_loss' in logs else 'loss' if 'loss' in logs else None
                    if fallback:
                        print(f"⚠️  Specified metric '{self.metric_name}' not found. Falling back to '{fallback}'")
                        self._initialize_metric_tracking(fallback, is_eval)
                        return logs[fallback], True
                    else:
                        if is_eval:
                            self._warn_missing_metric(self.metric_name)
                        elif self.verbose:
                            print(f"⚠️  Specified metric '{self.metric_name}' not found in logs. Available: {list(logs.keys())}")
                return None, False
            
            # Metric found - initialize if needed
            if self.detected_metric_name is None:
                self._initialize_metric_tracking(self.metric_name, is_eval)
            
            return logs[self.metric_name], True
        
        # Auto-detect metric
        detected = self._detect_metric_name(logs)
        if detected is None:
            if self.verbose:
                metric_type = "eval metrics" if is_eval else "logs"
                print(f"⚠️  No metric found in {metric_type}. Available: {list(logs.keys())}")
            return None, False
        
        # Initialize if needed
        if self.detected_metric_name is None:
            self.detected_metric_name = detected
            self.detected_greater_is_better = self._auto_detect_direction(detected)
            direction = "higher is better" if self.detected_greater_is_better else "lower is better"
            metric_type = "eval" if is_eval else "training"
            print(f"✓ Auto-detected {metric_type} metric: {detected} ({direction})")
        
        return logs[detected], True
    
    def _warn_missing_metric(self, metric_name):
        """Print a warning about missing metric only once."""
        if metric_name not in self._warned_metrics:
            print(f"\n⚠️  WARNING: Specified metric '{metric_name}' not found in evaluation metrics.")
            print(f"   This usually means:")
            print(f"   1. Your compute_metrics function isn't returning this metric")
            print(f"   2. Evaluation isn't running (check evaluation_strategy in TrainingArguments)")
            print(f"   3. The metric name doesn't match exactly (case-sensitive)")
            print(f"   Falling back to training loss monitoring.\n")
            self._warned_metrics.add(metric_name)
    
    def get_stop_reason(self):
        """Return a detailed description of why this callback stopped training."""
        if self.stopped_manually:
            return "EarlyStoppingCallback: Manual stop file detected"
        elif self.stopped_early:
            metric_info = f" ({self.detected_metric_name})" if self.detected_metric_name else ""
            return f"EarlyStoppingCallback{metric_info}: No improvement for {self.wait_count} checks"
        return None
    
    def _check_manual_stop(self, state: TrainerState, control: TrainerControl):
        """Check for manual stop file and handle stopping if found."""
        if not os.path.exists(self.stop_file_path):
            return False
        
        self.stopped_manually = True
        self.stop_reason = self.get_stop_reason()
        set_callback_stop_reason(self.stop_reason)
        control.should_training_stop = True
        
        print(f"\n{'='*80}")
        print(f"🛑 TRAINING STOPPED BY: EarlyStoppingCallback (Manual Stop)")
        print(f"{'='*80}")
        print(f"Reason: Manual stop file detected")
        print(f"Step: {state.global_step}")
        print(f"Stop file: {self.stop_file_path}")
        print(f"{'='*80}")
        
        try:
            os.remove(self.stop_file_path)
            print(f"Stop file removed successfully")
        except Exception as e:
            print(f"Warning: Could not remove stop file: {e}")
        
        return True
    
    def _check_metric_improvement(self, current_value, state: TrainerState, control: TrainerControl):
        """Check if metric improved and update counters. Returns True if training should stop."""
        self.check_count += 1
        
        # Initialize best_metric on first check
        if self.best_metric is None:
            self.best_metric = current_value
            self.best_step = state.global_step
            if self.verbose:
                direction = "higher is better" if self.detected_greater_is_better else "lower is better"
                print(f"✓ Step {state.global_step}: {self.detected_metric_name} initialized to {current_value:.4f} ({direction})")
            return False
        
        # Check for improvement based on direction
        if self.detected_greater_is_better:
            improved = current_value > self.best_metric + self.min_delta
        else:
            improved = current_value < self.best_metric - self.min_delta
        
        if improved:
            self.best_metric = current_value
            self.best_step = state.global_step
            self.wait_count = 0
            if self.verbose:
                print(f"✓ Step {state.global_step}: {self.detected_metric_name} improved to {current_value:.4f}")
            return False
        
        # No improvement
        self.wait_count += 1
        if self.verbose:
            print(f"  Step {state.global_step}: {self.detected_metric_name} {current_value:.4f} (no improvement for {self.wait_count}/{self.patience} checks)")
        
        if self.wait_count >= self.patience:
            self.stopped_early = True
            self.stop_reason = self.get_stop_reason()
            set_callback_stop_reason(self.stop_reason)
            control.should_training_stop = True
            
            direction = "higher is better" if self.detected_greater_is_better else "lower is better"
            print(f"\n{'='*80}")
            print(f"🛑 TRAINING STOPPED BY: EarlyStoppingCallback")
            print(f"{'='*80}")
            print(f"Reason: No improvement for {self.wait_count} consecutive logging checks (patience: {self.patience})")
            print(f"Metric: {self.detected_metric_name} ({direction})")
            print(f"Current step: {state.global_step}")
            print(f"Current value: {current_value:.4f}")
            print(f"Best value: {self.best_metric:.4f} (at step {self.best_step})")
            print(f"{'='*80}")
            return True
        
        return False
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize at training start."""
        metric_info = f"metric='{self.metric_name}'" if self.metric_name else "auto-detect"
        direction_info = ""
        if self.metric_name and self.greater_is_better is not None:
            direction_info = f" ({'higher' if self.greater_is_better else 'lower'} is better)"
        
        print(f"🎯 Early stopping enabled: patience={self.patience} logging checks, min_delta={self.min_delta}")
        print(f"    Metric tracking: {metric_info}{direction_info}")
        print(f"    Metric check every {args.logging_steps} steps")
        print(f"    Will stop if no improvement for {self.patience} consecutive logging checks")
        print(f"    Manual stop: Create '{self.stop_file_path}' to stop training")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Check metric at each logging step (for training metrics)."""
        if self._check_manual_stop(state, control) or logs is None:
            return control
        
        metric_value, should_continue = self._get_metric_value(logs, is_eval=False)
        if should_continue:
            self._check_metric_improvement(metric_value, state, control)
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check eval metrics at each evaluation step."""
        if self._check_manual_stop(state, control):
            return control
        
        metrics = kwargs.get("metrics")
        if metrics is None:
            return control
        
        metric_value, should_continue = self._get_metric_value(metrics, is_eval=True)
        if should_continue:
            self._check_metric_improvement(metric_value, state, control)
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Print summary at training end."""
        if self.stopped_manually:
            status = "stopped manually"
        elif self.stopped_early:
            status = "stopped early"
        else:
            status = "completed"
        
        direction = ""
        if self.detected_greater_is_better is not None:
            direction = f" ({'higher' if self.detected_greater_is_better else 'lower'} is better)"
        
        print(f"\n📊 Training {status}")
        print(f"   Metric: {self.detected_metric_name}{direction}")
        print(f"   Best value: {self.best_metric:.4f} at step {self.best_step}")
        print(f"   Total checks: {self.check_count}")



class TimeLimitCallback(TrainerCallback):
    """Stops training gracefully after a specified time limit.
    
    Args:
        max_hours (float): Maximum training time in hours. Default: 2.0
    """
    def __init__(self, max_hours=2.0):
        self.max_hours = max_hours
        self.max_seconds = max_hours * 3600
        self.start_time = None
        self.stopped_by_time_limit = False
        self.stop_reason = None  # Detailed stop reason for logging
    
    def get_stop_reason(self):
        """Return a detailed description of why this callback stopped training."""
        if self.stopped_by_time_limit:
            elapsed_hours = (time.time() - self.start_time) / 3600 if self.start_time else 0
            return f"TimeLimitCallback(max_hours={self.max_hours}): Time limit reached after {elapsed_hours:.2f} hours"
        return None
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Record start time."""
        self.start_time = time.time()
        print(f"🕑 Time limit enabled. Set to maximum of {self.max_hours} hours")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check elapsed time after each step."""
        if self.start_time and time.time() - self.start_time >= self.max_seconds:
            self.stopped_by_time_limit = True
            self.stop_reason = self.get_stop_reason()
            set_callback_stop_reason(self.stop_reason)  # Share with logger
            elapsed_hours = (time.time() - self.start_time) / 3600
            print(f"\n{'='*80}")
            print(f"🛑 TRAINING STOPPED BY: TimeLimitCallback")
            print(f"{'='*80}")
            print(f"Reason: Time limit reached")
            print(f"Time limit: {self.max_hours:.2f} hours")
            print(f"Elapsed time: {elapsed_hours:.2f} hours")
            print(f"Current step: {state.global_step}")
            print(f"{'='*80}")
            control.should_training_stop = True
        return control




class TrainingLoggerCallback(TrainerCallback):
    """Logs training metrics to text and JSON files.
    
    Automatically logs all training parameters from the trainer (learning rate, batch size,
    weight decay, lr scheduler, etc.). Optionally accepts additional model-specific config.
    
    Args:
        base_path (str): Directory where log files will be saved. Default: "."
        extra_config (dict): Optional additional config (e.g., LoRA params, max_seq_length). Default: None
    """
    def __init__(self, base_path=".", extra_config=None):
        self.base_path = base_path
        self.log_file = os.path.join(base_path, "training_log.txt")
        self.json_file = os.path.join(base_path, "training_log.json")
        self.extra_config = extra_config or {}
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_step = 0
        self.start_time = None
        self.stop_reason = None  # Track why training stopped
        
        # Memory tracking
        self.memory_history = []
        self.peak_ram_mb = 0
        self.peak_vram_mb = 0
        self.initial_ram_mb = 0
        self.initial_vram_mb = 0
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
        # CPU tracking
        self.cpu_history = []
        self.peak_cpu_percent = 0
        self.initial_cpu_percent = 0
    
    def _get_memory_usage(self):
        """Get current RAM and VRAM usage in MB."""
        ram_mb = 0
        vram_mb = 0
        
        # Get system RAM usage
        if PSUTIL_AVAILABLE and self.process:
            ram_mb = self.process.memory_info().rss / (1024 ** 2)
        
        # Get GPU VRAM usage
        if TORCH_AVAILABLE and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        return ram_mb, vram_mb
    
    def _get_cpu_usage(self):
        """Get current CPU utilization as percentage."""
        cpu_percent = 0
        
        # Get CPU usage (interval=None for non-blocking call after first measurement)
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=None)
        
        return cpu_percent
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Record start time and ensure output directory exists."""
        self.start_time = time.time()
        os.makedirs(self.base_path, exist_ok=True)
        
        # Clear any previous stop reason at the start of training
        clear_callback_stop_reason()
        
        # Initialize CPU monitoring (first call with interval to set baseline)
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=0.1)
        
        # Record initial memory and CPU usage
        self.initial_ram_mb, self.initial_vram_mb = self._get_memory_usage()
        self.initial_cpu_percent = self._get_cpu_usage()
        self.peak_ram_mb = self.initial_ram_mb
        self.peak_vram_mb = self.initial_vram_mb
        self.peak_cpu_percent = self.initial_cpu_percent
        
        print(f"📝 Training logger enabled. Logs will be saved to {self.base_path}")
        print(f"💾 Initial Memory - RAM: {self.initial_ram_mb:.2f} MB, VRAM: {self.initial_vram_mb:.2f} MB")
        print(f"⚙️ Initial CPU Usage: {self.initial_cpu_percent:.1f}%")
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Record loss, memory usage, and CPU usage at each logging step."""
        if logs is None or logs.get("loss") is None:
            return control
        
        current_loss = logs["loss"]
        
        # Get current memory and CPU usage
        ram_mb, vram_mb = self._get_memory_usage()
        cpu_percent = self._get_cpu_usage()
        
        # Update peaks
        self.peak_ram_mb = max(self.peak_ram_mb, ram_mb)
        self.peak_vram_mb = max(self.peak_vram_mb, vram_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
        
        # Record history
        self.loss_history.append({"step": state.global_step, "loss": current_loss})
        self.memory_history.append({
            "step": state.global_step,
            "ram_mb": ram_mb,
            "vram_mb": vram_mb
        })
        self.cpu_history.append({
            "step": state.global_step,
            "cpu_percent": cpu_percent
        })
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_step = state.global_step
        
        return control
    
    def _collect_stop_reasons_from_callbacks(self, **kwargs):
        """Collect stop reasons from all callbacks that have the get_stop_reason method."""
        # First check the global stop reason
        global_reason = get_callback_stop_reason()
        if global_reason:
            return global_reason
        
        # Try to get callbacks from kwargs
        callbacks = []
        
        # Check if we have direct access to callbacks through kwargs
        if 'callbacks' in kwargs:
            callbacks = kwargs['callbacks']
        
        # Try to get from model.trainer
        model = kwargs.get('model', None)
        if model and hasattr(model, 'trainer') and hasattr(model.trainer, 'callback_handler'):
            callbacks = model.trainer.callback_handler.callbacks
        
        # Collect stop reasons
        for callback in callbacks:
            if hasattr(callback, 'get_stop_reason') and callback is not self:
                reason = callback.get_stop_reason()
                if reason:
                    return reason
        
        return None
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Write log files at training end."""
        training_duration = time.time() - self.start_time
        
        # Determine stop reason by checking other callbacks
        if not self.stop_reason:
            # Try to collect stop reason from other callbacks
            collected_reason = self._collect_stop_reasons_from_callbacks(**kwargs)
            if collected_reason:
                self.stop_reason = collected_reason
            # If still no reason found, check if training was stopped early
            elif state.global_step < args.max_steps or (args.max_steps <= 0 and state.epoch < args.num_train_epochs):
                self.stop_reason = "Training stopped early (check callback messages above)"
            else:
                self.stop_reason = "Training completed normally"
        
        # Clear the global stop reason for next training run
        clear_callback_stop_reason()
        
        # Get final memory and CPU usage
        final_ram_mb, final_vram_mb = self._get_memory_usage()
        final_cpu_percent = self._get_cpu_usage()
        
        # Extract datasets from kwargs if available
        train_dataset = kwargs.get('train_dataloader', None)
        eval_dataset = kwargs.get('eval_dataloader', None)
        if train_dataset is not None and hasattr(train_dataset, 'dataset'):
            train_dataset = train_dataset.dataset

        if eval_dataset is not None and hasattr(eval_dataset, 'dataset'):
            eval_dataset = eval_dataset.dataset
        
        # Build text log
        log_lines = [
            "=" * 80,
            "TRAINING LOG",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {training_duration/3600:.2f} hours ({training_duration/60:.2f} minutes)",
            f"Stop Reason: {self.stop_reason}",
            "",
            "-" * 80,
            "TRAINING PARAMETERS",
            "-" * 80,
        ]
        
        # Add dataset info if available
        if train_dataset is not None and hasattr(train_dataset, '__len__'):
            log_lines.append(f"Train Dataset Size: {len(train_dataset)}")

        if eval_dataset is not None and hasattr(eval_dataset, '__len__'):
            log_lines.append(f"Eval Dataset Size: {len(eval_dataset)}")
        
        # Core training parameters
        log_lines.extend([
            f"Learning Rate: {args.learning_rate}",
            f"Batch Size (per device): {args.per_device_train_batch_size}",
            f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}",
            f"Effective Batch Size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}",
            f"Num Epochs: {args.num_train_epochs}",
            f"Warmup Steps: {args.warmup_steps}",
            f"Logging Steps: {args.logging_steps}",
        ])
        
        # Optimizer and scheduler
        if hasattr(args, 'optim'):
            log_lines.append(f"Optimizer: {args.optim}")
        
        log_lines.append(f"Weight Decay: {args.weight_decay}")
        log_lines.append(f"LR Scheduler: {args.lr_scheduler_type}")
        
        # Additional parameters
        if hasattr(args, 'seed'):
            log_lines.append(f"Seed: {args.seed}")
        
        if hasattr(args, 'report_to'):
            log_lines.append(f"Report To: {args.report_to}")
        
        if hasattr(args, 'torch_compile'):
            log_lines.append(f"Torch Compile: {args.torch_compile}")
        
        log_lines.extend([
            f"Max Grad Norm: {args.max_grad_norm}",
            f"Max Steps: {args.max_steps if args.max_steps > 0 else 'N/A'}",
            ""
        ])
        
        if self.extra_config:
            log_lines.extend([
                "-" * 80,
                "EXTRA CONFIGURATION",
                "-" * 80
            ])
            log_lines.extend([f"{k}: {v}" for k, v in self.extra_config.items()])
            log_lines.append("")
        
        log_lines.extend([
            "-" * 80,
            "SYSTEM RESOURCE USAGE",
            "-" * 80,
            "Memory:",
            f"  Initial RAM: {self.initial_ram_mb:.2f} MB",
            f"  Final RAM: {final_ram_mb:.2f} MB",
            f"  Peak RAM: {self.peak_ram_mb:.2f} MB",
            f"  RAM Increase: {final_ram_mb - self.initial_ram_mb:.2f} MB",
            "",
            f"  Initial VRAM: {self.initial_vram_mb:.2f} MB",
            f"  Final VRAM: {final_vram_mb:.2f} MB",
            f"  Peak VRAM: {self.peak_vram_mb:.2f} MB",
            f"  VRAM Increase: {final_vram_mb - self.initial_vram_mb:.2f} MB",
            "",
            "CPU:",
            f"  Initial CPU: {self.initial_cpu_percent:.1f}%",
            f"  Final CPU: {final_cpu_percent:.1f}%",
            f"  Peak CPU: {self.peak_cpu_percent:.1f}%",
            ""
        ])
        
        log_lines.extend([
            "-" * 80,
            "TRAINING SUMMARY",
            "-" * 80,
            f"Total Steps: {state.global_step}",
            f"Best Loss: {self.best_loss:.6f}",
            f"Best Step: {self.best_step}"
        ])
        
        if self.loss_history:
            initial_loss = self.loss_history[0]["loss"]
            final_loss = self.loss_history[-1]["loss"]
            improvement = initial_loss - final_loss
            improvement_pct = (improvement / initial_loss * 100) if initial_loss != 0 else 0.0
            log_lines.extend([
                f"Initial Loss: {initial_loss:.6f}",
                f"Final Loss: {final_loss:.6f}",
                f"Loss Improvement: {improvement:.6f} ({improvement_pct:.2f}%)"
            ])
        
        log_lines.extend([
            "",
            "-" * 80,
            "LOSS HISTORY",
            "-" * 80,
            f"{'Step':<10} {'Loss':<15}",
            "-" * 80
        ])
        
        for entry in self.loss_history:
            marker = " ← BEST" if entry["step"] == self.best_step else ""
            log_lines.append(f"{entry['step']:<10} {entry['loss']:<15.6f}{marker}")
        
        # Add resource usage history if available
        if self.memory_history and self.cpu_history:
            log_lines.extend([
                "",
                "-" * 80,
                "RESOURCE USAGE HISTORY",
                "-" * 80,
                f"{'Step':<10} {'RAM (MB)':<15} {'VRAM (MB)':<15} {'CPU (%)':<10}",
                "-" * 80
            ])
            
            for mem_entry, cpu_entry in zip(self.memory_history, self.cpu_history):
                log_lines.append(
                    f"{mem_entry['step']:<10} "
                    f"{mem_entry['ram_mb']:<15.2f} "
                    f"{mem_entry['vram_mb']:<15.2f} "
                    f"{cpu_entry['cpu_percent']:<10.1f}"
                )
        
        log_lines.append("=" * 80)
        
        # Write text log
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_lines))
        
        # Write JSON log
        training_params = {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "warmup_steps": args.warmup_steps,
            "logging_steps": args.logging_steps,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_grad_norm": args.max_grad_norm,
            "max_steps": args.max_steps if args.max_steps > 0 else None,
        }
        
        if train_dataset is not None and hasattr(train_dataset, '__len__'):
            training_params["train_dataset_size"] = len(train_dataset)

        if eval_dataset is not None and hasattr(eval_dataset, '__len__'):
            training_params["eval_dataset_size"] = len(eval_dataset)
        
        if hasattr(args, 'optim'):
            training_params["optim"] = args.optim
        
        if hasattr(args, 'seed'):
            training_params["seed"] = args.seed
        
        if hasattr(args, 'report_to'):
            training_params["report_to"] = args.report_to
        
        if hasattr(args, 'torch_compile'):
            training_params["torch_compile"] = args.torch_compile
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "training_duration_seconds": training_duration,
            "stop_reason": self.stop_reason,
            "training_parameters": training_params,
            "extra_config": self.extra_config,
            "resource_usage": {
                "memory": {
                    "initial_ram_mb": self.initial_ram_mb,
                    "final_ram_mb": final_ram_mb,
                    "peak_ram_mb": self.peak_ram_mb,
                    "ram_increase_mb": final_ram_mb - self.initial_ram_mb,
                    "initial_vram_mb": self.initial_vram_mb,
                    "final_vram_mb": final_vram_mb,
                    "peak_vram_mb": self.peak_vram_mb,
                    "vram_increase_mb": final_vram_mb - self.initial_vram_mb,
                },
                "cpu": {
                    "initial_cpu_percent": self.initial_cpu_percent,
                    "final_cpu_percent": final_cpu_percent,
                    "peak_cpu_percent": self.peak_cpu_percent,
                }
            },
            "summary": {
                "total_steps": state.global_step,
                "best_loss": self.best_loss,
                "best_step": self.best_step,
                "initial_loss": self.loss_history[0]["loss"] if self.loss_history else None,
                "final_loss": self.loss_history[-1]["loss"] if self.loss_history else None,
            },
            "loss_history": self.loss_history,
            "memory_history": self.memory_history,
            "cpu_history": self.cpu_history
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n📊 Training logs saved:")
        print(f"   Text: {self.log_file}")
        print(f"   JSON: {self.json_file}")
        print(f"   Best loss: {self.best_loss:.6f} at step {self.best_step}")
        print(f"💾 Memory Usage:")
        print(f"   RAM: {self.initial_ram_mb:.2f} MB → {final_ram_mb:.2f} MB (Peak: {self.peak_ram_mb:.2f} MB)")
        print(f"   VRAM: {self.initial_vram_mb:.2f} MB → {final_vram_mb:.2f} MB (Peak: {self.peak_vram_mb:.2f} MB)")
        print(f"⚙️  CPU Usage:")
        print(f"   CPU: {self.initial_cpu_percent:.1f}% → {final_cpu_percent:.1f}% (Peak: {self.peak_cpu_percent:.1f}%)")
        
        return control
