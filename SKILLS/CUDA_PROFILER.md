# CUDA PROFILER SKILL
## Running `ncu` (Nsight Compute) from a conda environment on Linux

---

## The Goal

Profile a CUDA kernel written with `load_inline` (PyTorch JIT extension) using
`ncu` to collect hardware metrics (DRAM throughput, sectors/request, SM utilization).

---

## Problems Encountered and How Each Was Fixed

### Problem 1 — `ncu` used the wrong Python (no `torch`)

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
==WARNING== No kernels were profiled.
```

**Why:** `ncu` spawns a fresh subprocess. That subprocess does NOT inherit the
conda environment — it uses the system Python from `/usr/bin/python` which has
no packages.

**Fix:** Pass the full path to the conda Python explicitly:

```bash
PYTHON=/path/to/project/.conda/bin/python
ncu ... $PYTHON my_script.py
```

---

### Problem 2 — `ninja` not found inside the `ncu` subprocess

**Symptom:**
```
RuntimeError: Ninja is required to load C++ extensions
==WARNING== No kernels were profiled.
```

**Why:** `load_inline` uses `ninja` to compile the CUDA kernel on first run.
`ninja` lives in the conda bin directory. The `ncu` subprocess inherits a
minimal PATH and cannot find it.

**Fix:** Export the conda bin directory onto PATH before calling `ncu`:

```bash
CONDA_BIN=/path/to/project/.conda/bin
export PATH="$CONDA_BIN:$PATH"
```

This must happen in the shell **before** the `ncu` call, not inside the `ncu`
subprocess.

---

### Problem 3 — Linux kernel blocks hardware performance counters

**Symptom:**
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
==WARNING== No kernels were profiled.
```

**Why:** Linux sets `/proc/sys/kernel/perf_event_paranoid = 2` by default,
which restricts access to hardware performance counters to root only.

**Fix (temporary — resets on reboot):**
```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

**Fix (permanent — survives reboots):**
```bash
echo 'kernel.perf_event_paranoid=0' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

---

### Problem 4 — NVIDIA driver blocks non-root profiling (second lock)

**Symptom:** Same `ERR_NVGPUCTRPERM` error, even after setting
`perf_event_paranoid=0`.

**Why:** NVIDIA driver 418.43+ added its own separate profiling restriction:
`RmProfilingAdminOnly`. Check it with:

```bash
cat /proc/driver/nvidia/params | grep -i profil
# RmProfilingAdminOnly: 1   ← this is the second lock
```

This is independent of `perf_event_paranoid`. Both must be unlocked.

**Fix (immediate — no reboot, use `sudo ncu`):**
```bash
sudo ncu ... $PYTHON my_script.py
```

**Fix (permanent — disable admin restriction via modprobe, requires reboot):**
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
    | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo update-initramfs -u
# reboot
```

After reboot, `ncu` works as a normal user without `sudo`.

---

### Problem 5 — `sudo` strips PATH, so ninja is lost again

**Symptom:** After switching to `sudo ncu`, ninja error returns:
```
RuntimeError: Ninja is required to load C++ extensions
```

**Why:** `sudo` resets PATH to a minimal secure set (`/usr/bin`, `/bin`, etc.)
and drops the conda bin directory — even though `export PATH=...` was done
in the parent shell.

**Fix:** Use `sudo env PATH="$PATH"` to forward the current PATH into the
root process:

```bash
sudo env PATH="$PATH" ncu ... $PYTHON my_script.py
```

This passes the caller's PATH explicitly as an environment variable to sudo,
bypassing sudo's PATH sanitization.

---

## Final Working Pattern

```bash
# At the top of your profile script:
CONDA_BIN=/path/to/project/.conda/bin
PYTHON=$CONDA_BIN/python
export PATH="$CONDA_BIN:$PATH"   # so ninja/nvcc are found by ncu subprocess

# For each ncu call:
sudo env PATH="$PATH" ncu \
    --kernel-name "my_kernel" \
    --metrics "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,..." \
    --target-processes all \
    $PYTHON my_script.py 2>&1
```

The `sudo env PATH="$PATH"` combo is the key:
- `sudo` → grants hardware counter access (bypasses NVIDIA's RmProfilingAdminOnly)
- `env PATH="$PATH"` → preserves the conda bin dir so ninja and nvcc are visible

---

## One-Time Setup Checklist (do once per machine)

```
Step 1: set perf_event_paranoid to 0
  echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

Step 2: make it permanent
  echo 'kernel.perf_event_paranoid=0' | sudo tee /etc/sysctl.d/99-perf.conf
  sudo sysctl --system

Step 3 (optional): disable NVIDIA admin-only profiling permanently
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
      | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  sudo update-initramfs -u
  # reboot — after this, no sudo needed for ncu
```

If Step 3 is done and you've rebooted, the pattern simplifies to:

```bash
env PATH="$PATH" ncu \      # no sudo needed
    --kernel-name "my_kernel" \
    ...
    $PYTHON my_script.py 2>&1
```

---

## Diagnosis Commands

```bash
# Check if ncu is installed
which ncu && ncu --version

# Check what Python ncu will use
which python

# Check perf_event_paranoid
cat /proc/sys/kernel/perf_event_paranoid

# Check NVIDIA admin profiling lock
cat /proc/driver/nvidia/params | grep -i profil

# Check NVIDIA driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Test ncu raw (no grep) to see actual output
sudo env PATH="$PATH" ncu \
    --kernel-name "my_kernel" \
    --metrics "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" \
    --target-processes all \
    $PYTHON my_script.py 2>&1
```

---

## `ncu-ui` Crashes on Startup (Qt xcb plugin error)

**Symptom:**
```
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
libxcb-cursor.so.0: cannot open shared object file: No such file or directory
Application could not be initialized!
Aborted (core dumped)
```

**Why:** `ncu-ui` uses Qt6. Since Qt 6.5.0, the `xcb` platform plugin requires
`libxcb-cursor0`, which is not installed by default on many Linux systems.

**Fix:**
```bash
sudo apt install libxcb-cursor0
```

Then rerun `ncu-ui`. No reboot needed.

---

## Why `--target-processes all`

Without this flag, `ncu` only watches the top-level Python process PID.
`load_inline` compiles the kernel in a subprocess (ninja + nvcc spawn child
processes). The CUDA context and kernels are launched from a child process,
not the top-level Python. `--target-processes all` tells `ncu` to attach to
every process in the tree.

---

## Why the `grep` was removed

The original script piped ncu output through:
```bash
ncu ... 2>&1 | grep -E "(kernel_name|Metric|sectors|throughput|dram|bytes)" | head -40
```

This silently swallowed all output when any of the earlier problems were
present — making it impossible to see the actual error messages.

**Lesson:** When debugging `ncu`, always remove the `grep` first and look at
the raw output. Only add grep back once you've confirmed ncu is actually
producing metric data.
