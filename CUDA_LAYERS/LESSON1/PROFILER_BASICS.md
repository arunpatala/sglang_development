# PROFILER BASICS
## How to measure what your GPU kernel is actually doing

---

## The One-Sentence Version

A GPU kernel launch returns to Python immediately — the GPU runs asynchronously. To know how fast your kernel really is, and *why*, you need profiling tools that measure what happens inside the GPU hardware.

---

## Part 1 — Why You Can't Just Use `time.time()`

This is the single most common mistake when profiling GPU code:

```python
import time

x = torch.randn(1024*1024, device="cuda", dtype=torch.float16)

# WRONG — this measures kernel LAUNCH time, not kernel EXECUTION time
start = time.time()
y = mod.my_kernel(x)        # Python returns immediately — GPU is still working
end = time.time()
print(end - start)          # ~0.00001 seconds — meaningless
```

**Why it's wrong:**

```
Python thread                     GPU hardware
─────────────                     ────────────
start = time.time()
mod.my_kernel(x)  ──launches──►  [kernel queued in stream]
end = time.time()                 [kernel still running...]
                                  [kernel finishes here — too late]
```

Python and the GPU run **in parallel**. The kernel launch just drops work into a queue (CUDA stream) and returns. Python moves on before the GPU is even halfway done.

**The fix: CUDA Events**

```python
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)

s.record()              # drop a timestamp marker into the GPU stream
y = mod.my_kernel(x)   # kernel queued after the marker
e.record()              # drop end marker — also queued, runs after kernel

torch.cuda.synchronize()            # wait for GPU to process everything above
ms = s.elapsed_time(e)              # now reads the real GPU time
print(f"{ms:.3f} ms")              # accurate GPU execution time
```

CUDA Events are timestamps placed **inside** the GPU stream. `elapsed_time` measures the gap between them on the GPU clock, not the CPU clock.

---

## Part 2 — The Two Profiling Tools You'll Use

```
Tool                When to use it
────────────────────────────────────────────────────────────────────
CUDA Events         Timing a specific kernel — "how many ms?"
                    Fast, zero overhead, use in every benchmark

torch.profiler      Timeline view — "which op is slow?"
                    Shows all PyTorch + CUDA ops in order
                    Good for finding bottlenecks in training loops

ncu (Nsight Compute)  Hardware-level analysis — "why is it slow?"
                    Shows cache hit rate, DRAM bandwidth, warp efficiency
                    The ground truth for kernel optimization
```

---

## Part 3 — CUDA Events: The Right Way to Time Kernels

### The complete benchmark template

```python
import torch

def bench(fn, warmup=10, iters=200):
    """
    fn      — a zero-argument callable that runs the kernel
    warmup  — throw-away runs to stabilize GPU clocks + fill JIT caches
    iters   — how many timed runs to average over
    Returns — milliseconds per iteration (float)
    """
    # Warmup: GPU clocks ramp up on first use.
    # Without warmup, first few runs are slow → biased average.
    for _ in range(warmup):
        fn()

    # Sync before starting: make sure warmup is fully done.
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(iters):
        fn()            # all 200 launches queue up in the stream
    e.record()

    torch.cuda.synchronize()    # wait for all 200 to finish

    return s.elapsed_time(e) / iters   # ms per single run


# Example usage
x = torch.randn(1024 * 1024, device="cuda", dtype=torch.float16)

ms = bench(lambda: mod.my_kernel(x))
print(f"{ms:.3f} ms per call")
```

### Turning milliseconds into bandwidth

For memory-bound kernels (copy, scale, elementwise), the useful number is **GB/s** — how much of the GPU's memory bandwidth you're using.

```python
ms = bench(lambda: mod.copy_kernel(src, dst))

n_elements = src.numel()
bytes_per_element = src.element_size()    # 2 for fp16, 4 for fp32

# count both the read (src) and the write (dst)
total_bytes = n_elements * bytes_per_element * 2

gb_per_s = total_bytes / ms / 1e6    # bytes → GB, ms → s

# RTX 4060 Ti peak: 288 GB/s
print(f"{ms:.3f} ms  |  {gb_per_s:.1f} GB/s  |  {gb_per_s/288*100:.1f}% of peak")
```

```
element_size():
  torch.float16  → 2 bytes
  torch.float32  → 4 bytes
  torch.int8     → 1 byte
  torch.bfloat16 → 2 bytes
```

### The warmup effect — why it matters

```
Run #   GPU clock   Measured time
──────  ─────────   ─────────────
1       ~800 MHz    0.85 ms    ← GPU "idle" clock — misleadingly slow
2       ~1500 MHz   0.47 ms
3       ~2000 MHz   0.34 ms
4–10    ~2500 MHz   0.28 ms    ← stable "boost" clock
11+     ~2500 MHz   0.28 ms    ← this is what you should measure
```

Without warmup, you measure the kernel AND the clock ramp-up time.

---

## Part 4 — `ncu`: Looking Inside the GPU Hardware

`ncu` (Nsight Compute CLI) is NVIDIA's profiler. It runs your Python script but intercepts each kernel launch, makes the GPU execute it in a special profiling mode, and reports hardware counter values.

### The simplest usage

```bash
# Profile everything (slow — every kernel is profiled)
ncu python my_script.py

# Profile only one specific kernel by name
ncu --kernel-name "copy_scalar" python my_script.py

# Profile and save to file for GUI
ncu --set full -o my_report python my_script.py
ncu-ui my_report.ncu-rep     # opens the GUI
```

### Running ncu on a Python file (what profile.sh does)

```bash
ncu \
    --kernel-name "copy_vec8_f16" \        # only profile this kernel
    --metrics "metric1,metric2,metric3" \  # which hardware counters to read
    --target-processes all \               # needed when Python spawns subprocesses
    python ex1_4_vec_copy_f16.py
```

`--target-processes all` is needed because Python launches CUDA through a subprocess. Without it, ncu only watches the top-level Python process and misses the kernel.

### Why ncu is slow

ncu replays each kernel **multiple times** (once per metric group) because the GPU only has a limited number of hardware counters that can be read simultaneously. A kernel that takes 0.3 ms normally might take 50 ms to profile. This is expected — only use ncu for analysis, not benchmarking.

---

## Part 5 — The Key Metrics and What They Mean

### The five metrics in `profile.sh`

```
Metric name (ncu ID)                                          What it measures
────────────────────────────────────────────────────────────────────────────────────────────
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum                Total bytes read from L1→global
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum                Total bytes written to global mem
sm__throughput.avg.pct_of_peak_sustained_elapsed            SM utilization % of peak
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio   Sectors per memory request
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed      DRAM bandwidth % of peak
```

Let's understand each one.

---

## Part 6 — Memory Coalescing: The Most Important Metric

### What is a "sector"?

GPU global memory is accessed in units of **32 bytes** called sectors.  
A **cache line** is 128 bytes = 4 sectors.

When a warp (32 threads) all read memory at the same time, the GPU tries to serve all 32 reads with as few memory transactions as possible.

```
Ideal (coalesced):
  Thread 0 reads element 0
  Thread 1 reads element 1
  ...
  Thread 31 reads element 31
  → All 32 elements fit in one 128-byte cache line = 1 request = 4 sectors
  → sectors_per_request ≈ 1 (each sector fully used)

Bad (uncoalesced):
  Thread 0 reads element 0
  Thread 1 reads element 32
  Thread 2 reads element 64
  ...
  → Each thread is in a different cache line
  → 32 separate memory requests = 32 sectors
  → sectors_per_request ≈ 32 (32× more DRAM traffic than needed)
```

### Scalar vs vectorized comparison

```
Kernel               Load instruction    Width    sectors_per_request
──────────────────────────────────────────────────────────────────────
copy_scalar          LD (16-bit)         2 bytes  ~1    ← only 2 bytes of 32-byte sector used
copy_vec8_f16        LDG.128             16 bytes ~8    ← all 16 bytes of a 128-bit load
```

Wait — scalar has sectors_per_request ~1 but that's also bad. Here's the subtlety:

**A scalar fp16 kernel loads 2 bytes per thread but the GPU fetches a whole 32-byte sector.**  
Only 2 of those 32 bytes are used → 93% waste → even though sectors_per_request is "low", DRAM bandwidth is low because the SM is underutilized waiting for memory.

**A vectorized kernel loads 16 bytes per thread (float4 = 128 bits).**  
The full sector is used → DRAM can sustain much higher bandwidth.

### What to look for:

```
Metric                  Scalar kernel        Vectorized kernel    Meaning
──────────────────────────────────────────────────────────────────────────
sectors_per_request     ~1                   ~8                   bytes per request
gpu__dram_throughput    10–30%               75–95%               actual DRAM usage
sm__throughput          very low             higher               SM busy time
```

The vectorized kernel has *higher* sectors_per_request but *higher* DRAM throughput because:
- Each request pulls more data
- Fewer total requests means less overhead
- The DRAM is kept busy continuously

---

## Part 7 — Reading `ncu` Output

### Raw ncu output looks like this:

```
==PROF== Connected to process 12345
==PROF== Profiling "copy_scalar" - 0: 0%....50%....100% - 18 passes

  copy_scalar, Context 1, Stream 7
    Section: GPU Speed Of Light Throughput
    ─────────────────────────────────────────────
    Metric Name                              Metric Unit    Metric Value
    ─────────────────────────────────────────────────────────────────────
    gpu__dram_throughput.avg.pct_of_...      %              14.32
    sm__throughput.avg.pct_of_...            %              3.78
    l1tex__average_t_sectors_per_...ratio    sector/op      1.00
```

The `grep` in `profile.sh` filters this to just the lines with metric values:

```bash
ncu ... python script.py 2>&1 | grep -E "(kernel_name|Metric|sectors|throughput|dram|bytes)" | head -40
```

### Side-by-side comparison output from profile.sh:

```
========================================================
  SCALAR COPY — copy_scalar kernel (ex1_1)
========================================================
    gpu__dram_throughput ...   14.32 %
    sm__throughput ...          3.78 %
    sectors_per_request ...     1.00 sector/op

========================================================
  VECTORIZED COPY — copy_vec8_f16 kernel (ex1_4)
========================================================
    gpu__dram_throughput ...   87.54 %
    sm__throughput ...         23.41 %
    sectors_per_request ...     8.00 sector/op
```

The vectorized kernel uses **6× more DRAM bandwidth** with the same data size, because it keeps the memory bus continuously busy.

---

## Part 8 — `torch.profiler`: The Timeline View

`torch.profiler` is a higher-level profiler built into PyTorch. It gives you a timeline of which operations ran, in what order, and how long each took. Use it when you want to find bottlenecks in a training loop or inference pipeline.

### Basic usage

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

x = torch.randn(1024*1024, device="cuda", dtype=torch.float16)

with profile(
    activities=[
        ProfilerActivity.CPU,    # record CPU events (kernel launches, Python)
        ProfilerActivity.CUDA,   # record GPU events (actual kernel execution)
    ],
    record_shapes=True,          # show tensor shapes in output
) as prof:
    with record_function("my_kernel"):   # label this block in the output
        y = mod.my_kernel(x, 0.5)

# Print a summary table
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Reading the output table

```
---------------------------------  --------  --------  --------  --------
Name                               CPU time  CUDA time Self CPU  Self CUDA
---------------------------------  --------  --------  --------  --------
my_kernel                          120us     280us     8us       280us
aten::empty                         45us       0us     45us        0us
---------------------------------  --------  --------  --------  --------
```

```
Column          Meaning
──────────────────────────────────────────────────────────
CPU time        Time spent on CPU (kernel launch + Python overhead)
CUDA time       Time the GPU spent executing this op
Self CPU        CPU time excluding children (nested ops)
Self CUDA       GPU time excluding children
```

Key insight: **CPU time ≠ CUDA time**. A kernel can launch in 5 µs of CPU time but take 300 µs of GPU time. The GPU time is the real cost.

### Exporting to Chrome timeline (visual)

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        y = mod.my_kernel(x, 0.5)

# Export for chrome://tracing
prof.export_chrome_trace("trace.json")
```

Open `chrome://tracing` in Chrome, load `trace.json` → see a visual timeline of every op.

---

## Part 9 — The Roofline Model: Am I Memory-Bound or Compute-Bound?

Every kernel is limited by **one** of two things:

```
Memory-bound:    kernel spends most time waiting for data from DRAM
                 Adding more FLOPs wouldn't help — the bottleneck is data transfer
                 Example: elementwise ops (copy, scale, add)

Compute-bound:   kernel spends most time doing math (FP32/FP16 multiply-adds)
                 Adding more memory bandwidth wouldn't help — the bottleneck is ALUs
                 Example: large matrix multiplications (GEMMs)
```

### How to tell which one your kernel is

**Arithmetic Intensity** = FLOPs ÷ Bytes moved

```python
# Example: elementwise scale (y[i] = x[i] * 0.5)
n = 1024 * 1024           # number of elements
bytes_read  = n * 2       # fp16 read
bytes_write = n * 2       # fp16 write
total_bytes = bytes_read + bytes_write   # = 4 MB

flops = n * 1             # 1 multiply per element
                          # = 1M FLOPs

arithmetic_intensity = flops / total_bytes   # = 0.25 FLOPs/byte
```

For an RTX 4060 Ti:
```
Peak DRAM bandwidth:       288 GB/s
Peak FP16 compute:        ~130 TFLOPS

Roofline crossover:        130e12 / 288e9 ≈ 451 FLOPs/byte
```

Your kernel has 0.25 FLOPs/byte → **deeply memory-bound**.

```
                     compute-bound region
TFLOPS ─────────────────────────────╮
                                    │╲
                                    │  ╲
                              (roofline)╲
                                    │    ╲ memory-bound region
                                    │      ╲
0 ──────────────────────────────────┴───────────────► FLOPs/byte
                                  451
                              ^ your kernel is here (0.25)
```

**For memory-bound kernels**: optimize for bandwidth (vectorized loads, avoid redundant reads).  
**For compute-bound kernels**: optimize for FP16 math, use tensor cores.

---

## Part 10 — `sm__throughput` and What It Tells You

`sm__throughput.avg.pct_of_peak_sustained_elapsed` measures how busy the Streaming Multiprocessors are as a percentage of theoretical peak.

```
Low SM throughput (< 20%):
  The SM is stalling, waiting for memory (L2 cache miss, DRAM fetch).
  Your kernel is memory-latency limited — not enough parallelism to hide latency.

Medium SM throughput (20–60%):
  Partial memory hiding — some warps available to switch to while others wait.
  Better, but room to improve occupancy.

High SM throughput (> 60%):
  SM is mostly computing, not waiting.
  Either your kernel is compute-bound, or you have good memory access patterns.
```

For the scalar vs vectorized comparison:
```
copy_scalar:      sm__throughput ≈ 3–5%
  → SM spends 95% of time waiting for tiny 16-bit loads to arrive from DRAM

copy_vec8_f16:    sm__throughput ≈ 20–30%
  → SM is busier: each load brings 128 bits, so warp gets more data per stall
```

---

## Part 11 — `gpu__dram_throughput` and Memory Bandwidth

`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` tells you what fraction of peak DRAM bandwidth your kernel is actually using.

```
Low DRAM throughput (< 30%):
  Data is being served from L2 cache (good!) — or kernel is compute-bound.
  Or: kernel is inefficient and sends too few requests to keep DRAM busy.

High DRAM throughput (> 70%):
  DRAM is the bottleneck — kernel is memory-bound.
  This is what a well-optimized memory-bound kernel looks like.
```

For a copy kernel, you want DRAM throughput to be as high as possible — it means you're fully utilizing the memory bus.

```
copy_scalar:      gpu__dram_throughput ≈ 14%
  → Only 14% of 288 GB/s = 40 GB/s actually used
  → Wasted: GPU can't generate enough requests because each thread loads 2 bytes

copy_vec8_f16:    gpu__dram_throughput ≈ 87%
  → 87% of 288 GB/s = 250 GB/s actually used
  → Near peak: GPU generates enough requests to keep DRAM fully busy
```

---

## Part 12 — The `profile.sh` Script Explained Line by Line

```bash
#!/usr/bin/env bash
set -e          # stop on any error

# The five metrics we care about, comma-separated (no spaces)
METRICS="l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

MODE=${1:-all}   # first argument, default = "all"

# Profile the scalar kernel only
ncu \
    --kernel-name "copy_scalar" \     # only intercept this kernel, skip others
    --metrics "$METRICS" \            # only collect these 5 metrics
    --target-processes all \          # watch all child processes (needed for Python)
    python ex1_1_scalar_copy.py \     # run this script
    2>&1 |                            # merge stderr into stdout (ncu writes to stderr)
    grep -E "(copy_scalar|Metric|sectors|throughput|dram|bytes)" |  # keep relevant lines
    head -40                          # limit output to 40 lines
```

**Why `2>&1`?** — `ncu` writes its metric output to stderr. Without redirecting stderr to stdout, the pipe (`|`) would only see the Python script's stdout and miss all the ncu output.

**Why `--kernel-name`?** — Without it, ncu profiles every kernel in the process, including PyTorch internals. Filtering to one kernel gives focused output and runs faster.

**Why `grep`?** — Raw ncu output has many lines of headers, section titles, and separators. The grep filters to lines that contain the actual metric values.

---

## Part 13 — Common Profiling Mistakes and How to Avoid Them

### Mistake 1: Timing the first run

```python
# WRONG: first run includes JIT compilation time
ms = bench(fn, warmup=0, iters=1)

# RIGHT: warmup first
ms = bench(fn, warmup=10, iters=200)
```

### Mistake 2: Not synchronizing before reading results

```python
# WRONG: GPU might not be done
y = mod.my_kernel(x)
print(y.sum())       # may read partially written memory

# RIGHT
y = mod.my_kernel(x)
torch.cuda.synchronize()
print(y.sum())

# Also right: .item() forces sync automatically
print(y.sum().item())
```

### Mistake 3: Forgetting that ncu slows everything down

```python
# ncu replay makes kernels take 50–200x longer than normal
# Do NOT benchmark inside ncu
# Use ncu only for analysis, CUDA Events only for benchmarking
```

### Mistake 4: Measuring the wrong thing

```python
# WRONG: measuring Python overhead, not kernel time
ms = bench(lambda: (mod.my_kernel(x), torch.cuda.synchronize()))

# RIGHT: synchronize happens outside the timed region
ms = bench(lambda: mod.my_kernel(x))
torch.cuda.synchronize()   # after timing is done
```

### Mistake 5: Using too few iterations

```python
# WRONG: one fast kernel in 1 ms has ~1% variance from clock jitter
ms = bench(fn, iters=5)     # 5 samples → noisy

# RIGHT: average over many runs
ms = bench(fn, iters=200)   # noise averages out
```

---

## Part 14 — Putting It Together: Full Profiling Workflow

This is the recommended workflow when optimizing a kernel:

```
Step 1: Write the kernel
         ↓
Step 2: Verify correctness
         torch.testing.assert_close(output, reference)
         ↓
Step 3: Benchmark with CUDA Events
         ms = bench(lambda: mod.my_kernel(x))
         gb_per_s = bytes / ms / 1e6
         print(f"{gb_per_s:.1f} GB/s  ({gb_per_s/peak*100:.1f}% of peak)")
         ↓
Step 4: If below ~70% peak → profile with ncu
         bash profile.sh                     (quick metrics)
         ncu --set full -o report python ...  (full profile)
         ↓
Step 5: Read the key metrics
         sectors_per_request → are loads coalesced?
         dram_throughput     → is DRAM busy?
         sm_throughput       → is SM stalling?
         ↓
Step 6: Identify bottleneck → fix → repeat from Step 2
```

---

## Quick Reference

```
Task                                      Code
──────────────────────────────────────────────────────────────────────────────
time a kernel correctly                   CUDA Events + torch.cuda.synchronize()
benchmark template                        bench(fn, warmup=10, iters=200)
convert ms to GB/s                        (n_elem * elem_size * 2) / ms / 1e6
profile specific kernel                   ncu --kernel-name "name" python script.py
save ncu report to file                   ncu --set full -o report python script.py
open ncu GUI                              ncu-ui report.ncu-rep
timeline of all ops                       torch.profiler with ProfilerActivity.CUDA
export timeline to Chrome                 prof.export_chrome_trace("trace.json")
sectors_per_request → coalescing          low = each load wastes bandwidth
                                          high = each load fills the cache line
dram_throughput < 30%                     bad: DRAM underutilized
dram_throughput > 70%                     good: memory-bound, DRAM fully busy
sm_throughput < 10%                       SM stalling on memory (latency-bound)
arithmetic intensity                      FLOPs ÷ bytes_moved
memory-bound crossover (RTX 4060 Ti)      ~451 FLOPs/byte
peak DRAM bandwidth (RTX 4060 Ti)         288 GB/s
```
