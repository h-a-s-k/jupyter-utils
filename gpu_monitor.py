"""
Non-blocking System Monitor Widget for Jupyter
Displays GPU, CPU, RAM, and Disk usage with auto-refresh
"""

import subprocess
import threading
import time
import psutil
from IPython.display import display, HTML
import ipywidgets as widgets


class SystemMonitor:
    def __init__(self, refresh_interval=2.0):
        """
        Create a non-blocking system monitor widget.
        
        Args:
            refresh_interval (float): Update interval in seconds (default: 2.0)
        """
        self.refresh_interval = refresh_interval
        self.running = False
        self.thread = None
        
        # Detect GPU command
        self.gpu_cmd = self._detect_gpu_command()
        
        # Create output widget
        self.output = widgets.HTML(value=self._get_initial_html())
        
        # Create control buttons
        self.start_btn = widgets.Button(
            description='Start',
            button_style='success',
            icon='play'
        )
        self.stop_btn = widgets.Button(
            description='Stop',
            button_style='danger',
            icon='stop',
            disabled=True
        )
        self.refresh_slider = widgets.FloatSlider(
            value=refresh_interval,
            min=0.5,
            max=10.0,
            step=0.5,
            description='Refresh (s):',
            style={'description_width': '80px'}
        )
        
        # Button callbacks
        self.start_btn.on_click(lambda b: self.start())
        self.stop_btn.on_click(lambda b: self.stop())
        self.refresh_slider.observe(self._on_refresh_change, names='value')
        
        # Layout
        controls = widgets.HBox([self.start_btn, self.stop_btn, self.refresh_slider])
        self.widget = widgets.VBox([controls, self.output])
    
    def _detect_gpu_command(self):
        """Detect which GPU monitoring command is available."""
        commands = [
            ('rocm-smi', ['rocm-smi', '--showuse', '--showmeminfo', 'vram']),
            ('nvidia-smi', ['nvidia-smi']),
            ('amd-smi', ['amd-smi', 'static']),
        ]
        
        for name, cmd in commands:
            try:
                result = subprocess.run(
                    cmd[:1] + ['--help'],
                    capture_output=True,
                    timeout=1
                )
                if result.returncode == 0:
                    return (name, cmd)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return (None, None)
    
    def _parse_rocm_output(self, output):
        """Parse rocm-smi output into a clean format."""
        lines = output.strip().split('\n')
        gpu_data = {}
        
        for line in lines:
            if 'GPU use (%)' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    gpu_data['usage'] = parts[-1].strip()
            elif 'VRAM Total Memory (B)' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    total_bytes = int(parts[-1].strip())
                    gpu_data['vram_total'] = total_bytes / (1024**3)  # Convert to GB
            elif 'VRAM Total Used Memory (B)' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    used_bytes = int(parts[-1].strip())
                    gpu_data['vram_used'] = used_bytes / (1024**3)  # Convert to GB
        
        if not gpu_data:
            return None
        
        # Calculate VRAM percentage
        vram_percent = 0
        if 'vram_total' in gpu_data and 'vram_used' in gpu_data and gpu_data['vram_total'] > 0:
            vram_percent = (gpu_data['vram_used'] / gpu_data['vram_total']) * 100
        
        # Create progress bars
        bar_width = 30
        
        # GPU usage bar
        usage = int(gpu_data.get('usage', 0))
        usage_filled = int(bar_width * usage / 100)
        usage_bar = '█' * usage_filled + '░' * (bar_width - usage_filled)
        usage_color = '#4a4' if usage < 70 else '#fa0' if usage < 90 else '#f44'
        
        # VRAM bar
        vram_filled = int(bar_width * vram_percent / 100)
        vram_bar = '█' * vram_filled + '░' * (bar_width - vram_filled)
        vram_color = '#4a4' if vram_percent < 70 else '#fa0' if vram_percent < 90 else '#f44'
        
        html = f"""
        <div style='margin-bottom: 10px;'>
            <strong>GPU Usage:</strong> {usage}%
            <div style='font-family: monospace; color: {usage_color};'>{usage_bar} {usage}%</div>
        </div>
        """
        
        if 'vram_total' in gpu_data and 'vram_used' in gpu_data:
            html += f"""
        <div style='margin-bottom: 10px;'>
            <strong>VRAM:</strong> {gpu_data['vram_used']:.1f} GB / {gpu_data['vram_total']:.1f} GB ({vram_percent:.1f}%)
            <div style='font-family: monospace; color: {vram_color};'>{vram_bar} {vram_percent:.1f}%</div>
        </div>
        """
        
        return html
    
    def _get_gpu_info(self):
        """Get GPU information."""
        if not self.gpu_cmd[0]:
            return "<div style='color: #888;'>No GPU monitoring tool detected</div>"
        
        try:
            result = subprocess.run(
                self.gpu_cmd[1],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                # Parse ROCm output nicely
                if self.gpu_cmd[0] == 'rocm-smi':
                    parsed = self._parse_rocm_output(output)
                    if parsed:
                        return parsed
                
                # For nvidia-smi or unparseable output, show raw output
                output = output.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                return f"<pre style='margin: 0; font-size: 11px; line-height: 1.3;'>{output}</pre>"
            else:
                return f"<div style='color: #f44;'>GPU command failed</div>"
        except subprocess.TimeoutExpired:
            return "<div style='color: #f80;'>GPU command timeout</div>"
        except Exception as e:
            return f"<div style='color: #f44;'>Error: {str(e)}</div>"
    
    def _get_cpu_info(self):
        """Get CPU usage information."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        freq_str = f"{cpu_freq.current:.0f} MHz" if cpu_freq else "N/A"
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * cpu_percent / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        color = '#4a4' if cpu_percent < 70 else '#fa0' if cpu_percent < 90 else '#f44'
        
        return f"""
        <div style='margin-bottom: 10px;'>
            <strong>CPU Usage:</strong> {cpu_percent:.1f}% ({cpu_count} cores @ {freq_str})
            <div style='font-family: monospace; color: {color};'>{bar} {cpu_percent:.1f}%</div>
        </div>
        """
    
    def _get_memory_info(self):
        """Get RAM usage information."""
        mem = psutil.virtual_memory()
        
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        percent = mem.percent
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        color = '#4a4' if percent < 70 else '#fa0' if percent < 90 else '#f44'
        
        return f"""
        <div style='margin-bottom: 10px;'>
            <strong>RAM:</strong> {used_gb:.1f} GB / {total_gb:.1f} GB ({percent:.1f}%)
            <div style='font-family: monospace; color: {color};'>{bar} {percent:.1f}%</div>
        </div>
        """
    
    def _get_disk_info(self):
        """Get disk usage information."""
        disk = psutil.disk_usage('/')
        
        used_gb = disk.used / (1024**3)
        total_gb = disk.total / (1024**3)
        percent = disk.percent
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * percent / 100)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        color = '#4a4' if percent < 70 else '#fa0' if percent < 90 else '#f44'
        
        return f"""
        <div style='margin-bottom: 10px;'>
            <strong>Disk (/):</strong> {used_gb:.1f} GB / {total_gb:.1f} GB ({percent:.1f}%)
            <div style='font-family: monospace; color: {color};'>{bar} {percent:.1f}%</div>
        </div>
        """
    
    def _get_initial_html(self):
        """Get initial HTML content."""
        return """
        <div style='padding: 10px; background: #f5f5f5; border-radius: 5px; font-family: sans-serif;'>
            <div style='color: #888; text-align: center;'>Click Start to begin monitoring</div>
        </div>
        """
    
    def _update_display(self):
        """Update the display with current system info."""
        gpu_info = self._get_gpu_info()
        cpu_info = self._get_cpu_info()
        mem_info = self._get_memory_info()
        disk_info = self._get_disk_info()
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <div style='padding: 15px; background: #f5f5f5; border-radius: 5px; font-family: sans-serif;'>
            <div style='margin-bottom: 15px; color: #666; font-size: 12px;'>
                Last updated: {timestamp}
            </div>
            
            {cpu_info}
            {mem_info}
            {disk_info}
            
            <div style='margin-top: 15px;'>
                <div style='margin-bottom: 5px;'><strong>GPU ({self.gpu_cmd[0] or 'N/A'}):</strong></div>
                <div style='background: #fff; padding: 10px; border-radius: 3px; overflow-x: auto;'>
                    {gpu_info}
                </div>
            </div>
        </div>
        """
        
        self.output.value = html
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                self._update_display()
                time.sleep(self.refresh_interval)
            except Exception as e:
                print(f"Monitor error: {e}")
                break
    
    def _on_refresh_change(self, change):
        """Handle refresh interval change."""
        self.refresh_interval = change['new']
    
    def start(self):
        """Start monitoring."""
        if not self.running:
            self.running = True
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if self.running:
            self.running = False
            self.start_btn.disabled = False
            self.stop_btn.disabled = True
            if self.thread:
                self.thread.join(timeout=self.refresh_interval + 1)
    
    def display(self):
        """Display the widget."""
        display(self.widget)
        return self
    
    def __repr__(self):
        """String representation."""
        return f"SystemMonitor(refresh_interval={self.refresh_interval})"


# Convenience function
def monitor(refresh_interval=2.0, auto_start=True):
    """
    Create and display a system monitor widget.
    
    Args:
        refresh_interval (float): Update interval in seconds (default: 2.0)
        auto_start (bool): Start monitoring immediately (default: True)
    
    Returns:
        SystemMonitor: The monitor instance
    
    Example:
        >>> mon = monitor()
        >>> # Do other work in other cells
        >>> mon.stop()  # Stop when done
    """
    mon = SystemMonitor(refresh_interval=refresh_interval)
    mon.display()
    if auto_start:
        mon.start()
    return mon
