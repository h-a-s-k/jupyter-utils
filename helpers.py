"""
Jupyter Notebook Helper Functions for UV Virtual Environments

Provides clean utilities to:
1. Create a UV virtual environment and register it as a Jupyter kernel
2. Execute pip commands in the detected virtual environment
3. Execute arbitrary bash commands
"""

import sys
import subprocess
import os
import json


# ========== HELPER FUNCTIONS ==========

def bash(cmd, show_output=True):
    """
    Execute a bash command (replacement for !command in Jupyter).
    
    Args:
        cmd (str): The bash command to run
        show_output (bool): If True, prints output in real-time
    
    Returns:
        subprocess.CompletedProcess or subprocess.Popen
    """
    print(f"$ {cmd}")
    
    if show_output:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        return proc
    else:
        return subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )


def get_venv_root():
    """Detect the root directory of the current virtual environment."""
    bin_dir = os.path.dirname(sys.executable)
    venv_root = os.path.dirname(bin_dir)
    
    if os.path.basename(bin_dir) != "bin" or not os.path.exists(os.path.join(venv_root, "bin", "activate")):
        raise RuntimeError(
            f"Could not detect a valid virtual environment.\n"
            f"sys.executable: {sys.executable}\n"
            f"Expected a 'bin' directory with an 'activate' script."
        )
    return venv_root


def pip(cmd, show_output=True):
    """
    Execute a pip command in the current virtual environment.
    
    Args:
        cmd (str): The pip command (e.g., "install unsloth" or "list")
        show_output (bool): If True, prints output in real-time
    
    Returns:
        subprocess.CompletedProcess or subprocess.Popen
    """
    venv_root = get_venv_root()
    python_path = os.path.join(venv_root, "bin", "python")
    
    # Check if uv is available
    uv_paths = [
        os.path.expanduser("~/.local/bin/uv"),
        "/root/.local/bin/uv",
        "/usr/local/bin/uv"
    ]
    
    uv_cmd = None
    for uv_path in uv_paths:
        if os.path.exists(uv_path):
            uv_cmd = uv_path
            break
    
    if uv_cmd:
        # Use uv pip with --python flag after the subcommand
        # Extract the pip subcommand (install, list, etc.)
        parts = cmd.split(None, 1)
        if len(parts) > 1:
            subcommand = parts[0]
            rest = parts[1]
            full_cmd = f"{uv_cmd} pip {subcommand} --python {python_path} {rest}"
        else:
            full_cmd = f"{uv_cmd} pip {cmd} --python {python_path}"
        print(f"🐍 [{os.path.basename(venv_root)}] uv pip {cmd}")
    else:
        # Fall back to regular pip with explicit python -m pip
        full_cmd = f"{python_path} -m pip {cmd}"
        print(f"\n⚠️  Falling back to non-uv pip.")
        print(f"🐍⚠️ [{os.path.basename(venv_root)}] pip {cmd}")
    
    if show_output:
        proc = subprocess.Popen(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        return proc
    else:
        return subprocess.run(
            full_cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True
        )


# ========== UV ENVIRONMENT SETUP ==========

def create_uv_kernel(env_dir="uv_env", kernel_name="uv_env", 
                     display_name="UV ROCM Env", python_version="3.13"):
    """
    Create a UV virtual environment and register it as a Jupyter kernel.
    
    Args:
        env_dir (str): Directory name for the virtual environment
        kernel_name (str): Internal kernel identifier
        display_name (str): Name shown in Jupyter's kernel menu
        python_version (str): Python version to use (e.g., "3.13", "3.11")
    
    Returns:
        str: Path to the created virtual environment
    """
    # Ensure uv is installed
    uv_path = os.path.expanduser("~/.local/bin/uv")
    if not os.path.exists(uv_path):
        print("📦 Installing uv...")
        bash("curl -LsSf https://astral.sh/uv/install.sh | sh")
    else:
        print("✓ uv already installed")
    
    # Add uv to PATH
    os.environ["PATH"] = f"{os.environ.get('PATH', '')}:{os.path.expanduser('~/.local/bin')}"
    
    # Create virtual environment
    venv_path = os.path.abspath(env_dir)
    if os.path.exists(venv_path):
        print(f"🗑️  Removing existing environment at {venv_path}")
        bash(f"rm -rf {venv_path}")
    
    print(f"🔨 Creating virtual environment (Python {python_version})...")
    bash(f"uv venv {venv_path} --python {python_version}")
    
    # Install ipykernel
    print("📦 Installing ipykernel...")
    bash(f"source {venv_path}/bin/activate && uv pip install ipykernel")
    
    # Create kernel spec
    kernel_dir = os.path.expanduser(f"~/.local/share/jupyter/kernels/{kernel_name}")
    os.makedirs(kernel_dir, exist_ok=True)
    
    kernel_json = {
        "argv": [
            f"{venv_path}/bin/python",
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}"
        ],
        "display_name": display_name,
        "language": "python"
    }
    
    with open(os.path.join(kernel_dir, "kernel.json"), "w") as f:
        json.dump(kernel_json, f, indent=2)
    
    print(f"✓ Kernel spec created at {kernel_dir}")
    
    # Verify
    proc = bash("jupyter kernelspec list", show_output=False)
    if kernel_name in proc.stdout:
        print(f"\n✅ SUCCESS! Kernel '{display_name}' is ready.")
    else:
        print(f"\n⚠️  Kernel not found in list. Try refreshing the page.")
    
    print(f"\n👉 Next steps:")
    print(f"   1. Refresh this page")
    print(f"   2. Go to Kernel → Change kernel → '{display_name}'")
    print(f"   3. Verify with: import sys; print(sys.executable)")
    
    return venv_path


def remove_kernel(kernel_name):
    """
    Remove a Jupyter kernel and optionally its virtual environment.
    
    Args:
        kernel_name (str): The kernel name to remove (e.g., "uv_env")
    
    Returns:
        bool: True if successful, False otherwise
    """
    import shutil
    
    # Remove kernel spec
    kernel_dir = os.path.expanduser(f"~/.local/share/jupyter/kernels/{kernel_name}")
    
    if os.path.exists(kernel_dir):
        print(f"🗑️  Removing kernel spec: {kernel_dir}")
        shutil.rmtree(kernel_dir)
        print(f"✓ Kernel '{kernel_name}' removed")
    else:
        print(f"⚠️  Kernel '{kernel_name}' not found at {kernel_dir}")
    
    # Verify removal
    proc = bash("jupyter kernelspec list", show_output=False)
    if kernel_name not in proc.stdout:
        print(f"✅ Kernel '{kernel_name}' successfully removed from Jupyter")
        return True
    else:
        print(f"⚠️  Kernel '{kernel_name}' may still be listed")
        return False


def cleanup_session(kernel_name="uv_env", remove_venv=False):
    """
    Clean up at the end of a session: remove kernel and optionally the venv.
    
    Args:
        kernel_name (str): The kernel name to remove
        remove_venv (bool): If True, also removes the virtual environment directory
    
    Example:
        cleanup_session("uv_env", remove_venv=True)
    """
    import shutil
    
    print("🧹 Cleaning up session...")
    
    # Remove kernel
    remove_kernel(kernel_name)
    
    # Optionally remove venv directory
    if remove_venv:
        venv_path = os.path.abspath(kernel_name)
        if os.path.exists(venv_path):
            print(f"🗑️  Removing virtual environment: {venv_path}")
            shutil.rmtree(venv_path)
            print(f"✓ Virtual environment removed")
        else:
            print(f"⚠️  Virtual environment not found at {venv_path}")
    
    print("✅ Cleanup complete!")