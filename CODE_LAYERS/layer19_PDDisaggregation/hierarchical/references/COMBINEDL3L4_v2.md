# Chapter: Prefill-Decode Disaggregation in Large Language Model Serving

**Format:** Textbook-style prose chapter with pseudocode illustrations. Concepts build sequentially; each section establishes the ground for the next. Systems (DistServe, Splitwise, SARATHI, Mooncake, NIXL, Dynamo, TaiChi, vLLM) are cited as examples within a concept-driven narrative, not as the organizing principle. Pseudocode demonstrates each concept immediately after it is explained in prose.

---

## 1. The Two Phases of LLM Inference

To understand why the serving of large language models is architecturally challenging, we must begin with the most fundamental observation about how inference works: a single request to a language model involves two computationally distinct phases that have almost nothing in common from the perspective of hardware utilization.

### 1.1 The Prefill Phase

When a user submits a prompt, the model must first process the entire input sequence before it can generate a single output token. This processing step is called the **prefill phase**, and it is characterized by a single property that dominates everything else: the model can process all input tokens simultaneously, in parallel, in a single forward pass.

During prefill, the model computes three vectors — Query, Key, and Value — for every token in the input, across every attention layer. The Key and Value vectors for every token are then stored in GPU high-bandwidth memory (HBM) as the **KV cache**, which will serve as the model's working memory for the remainder of the request. The KV cache exists for a practical reason: without it, every subsequent decode step would need to recompute attention over the entire growing sequence from scratch, an operation whose cost scales quadratically with sequence length. By materializing and storing the K and V tensors once during prefill, every decode step can run in O(n) time rather than O(n²).

```
# Pseudocode: One attention layer during prefill
# input_tokens: shape [seq_len, d_model]  -- all tokens in parallel

function prefill_attention_layer(input_tokens, W_Q, W_K, W_V, W_O):
    # Project all input tokens to Q, K, V simultaneously
    Q = input_tokens @ W_Q          # shape [seq_len, d_head]
    K = input_tokens @ W_K          # shape [seq_len, d_head]
    V = input_tokens @ W_V          # shape [seq_len, d_head]

    # Attend over all token positions
    scores = Q @ K.T / sqrt(d_head) # shape [seq_len, seq_len]
    scores = causal_mask(scores)    # future tokens masked to -inf
    attn   = softmax(scores)        # shape [seq_len, seq_len]
    output = attn @ V               # shape [seq_len, d_head]

    # *** Store K, V permanently in HBM -- this IS the KV cache ***
    kv_cache.append(layer_id, K, V)

    return output @ W_O

# After the forward pass over all layers, kv_cache holds
# [n_layers x seq_len x d_head] Key and Value tensors.
# Prefill ends here. No output token has been generated yet.
```

From a hardware perspective, the prefill phase is dominated by large matrix multiplications that multiply the full sequence length against the model's weight matrices. For a 4,096-token prompt processed by a model like LLaMA-3.1-70B, the arithmetic intensity — defined as floating-point operations per byte of memory accessed — sits in the range of 200 to 400 FLOP per byte. This value places prefill far to the right of the roofline model's ridge point, firmly in the compute-bound regime. The GPU's tensor cores operate at 90 to 95 percent utilization. The HBM bus, by contrast, is barely touched — the computation requires only the model weights, which are accessed once and reused across all input tokens.

### 1.2 The Decode Phase

After prefill completes, the model enters the **decode phase**, in which it generates output tokens one at a time, sequentially. This sequential constraint is not an engineering choice that can be optimized away — it is a mathematical property of autoregressive generation. Each output token depends on all previous output tokens, which means the model cannot begin computing token $t+1$ until it knows the value of token $t$.

In each decode step, the model reads the entire KV cache from HBM, attends over the full accumulated sequence, generates a probability distribution over the vocabulary, samples the next token, appends its Key and Value vectors to the cache, and repeats. The dominant cost of each step is reading the KV cache, and that cache grows by one row with every token generated.

```
# Pseudocode: Autoregressive decode loop
# kv_cache: built during prefill, grows with every generated token

function decode(kv_cache, W_Q, W_K, W_V, W_O, W_lm_head, max_new_tokens):
    generated = []

    for step in range(max_new_tokens):
        last_token = get_last_token(generated or prefill_output)

        # Each step operates on ONE new token (batch_size=1 per request)
        q = last_token @ W_Q        # shape [1, d_head]
        k = last_token @ W_K        # shape [1, d_head]
        v = last_token @ W_V        # shape [1, d_head]

        # *** Read entire KV cache from HBM on every single step ***
        all_K = kv_cache.get_all_keys()    # shape [step + seq_len, d_head]
        all_V = kv_cache.get_all_values()  # shape [step + seq_len, d_head]

        scores = q @ all_K.T / sqrt(d_head)  # attend over all positions
        attn   = softmax(scores)
        output = attn @ all_V                # shape [1, d_head]

        logits    = output @ W_lm_head       # shape [1, vocab_size]
        next_token = sample(logits)

        # Append new K, V to cache -- it grows by one row each step
        kv_cache.append(k, v)
        generated.append(next_token)

        if next_token == EOS: break

    return generated
```

The arithmetic intensity of each decode step is approximately 1 FLOP per byte, regardless of sequence length. This is because the dominant operation — reading the KV cache — is pure memory access with minimal reuse of the loaded data. The GPU's tensor cores operate at 20 to 40 percent utilization even with large decode batches, because autoregressive generation cannot produce enough arithmetic to keep them busy. What limits decode throughput is not compute — it is the rate at which HBM can deliver KV cache data to the attention kernels.

### 1.3 Why the Asymmetry Is Not Incidental

It is tempting to treat the difference between prefill and decode as a matter of degree. This framing misses how severe the asymmetry actually is. Consider the arithmetic intensity calculation for each phase:

```
# Pseudocode: Arithmetic intensity for a single attention layer

# PREFILL: operating on a full sequence of S tokens
S          = 4096      # tokens in prompt
d_model    = 8192      # LLaMA-70B hidden dimension
d_head     = 128
n_heads    = 64
n_kv_heads = 8         # grouped-query attention

# FLOPs: Q, K, V projections + QK^T matmul + AV matmul
flops_prefill = (
    S * d_model * d_head * 3 +   # Q, K, V projections
    S * S * d_head +              # QK^T attention scores
    S * S * d_head                # AV weighted sum
)                                 # ≈ 2 * S^2 * d + 6 * S * d^2 (full sequence)

# Bytes: load weight matrices once; load/store KV cache
bytes_prefill = d_model * d_head * 3 * 2   # weight bytes (FP16)
# Weights are reused across all S tokens -- memory access is minimal

arithmetic_intensity_prefill = flops_prefill / bytes_prefill
# ≈ 200-400 FLOP/byte for S=4096
# → GPU is compute-bound, HBM bus underutilised

# DECODE: operating on ONE new token per step
flops_decode = (
    1 * d_model * d_head * 3 +   # Q, K, V for 1 token
    S * d_head +                  # q @ K^T (dot product over all cached keys)
    S * d_head                    # a @ V   (weighted sum over all cached values)
)

# Bytes: must READ the entire KV cache from HBM every step
bytes_decode = S * n_kv_heads * d_head * 2 * 2   # all K + all V, FP16

arithmetic_intensity_decode = flops_decode / bytes_decode
# ≈ 1 FLOP/byte regardless of S
# → GPU is memory-bandwidth-bound, tensor cores underutilised
```

Splitwise validated this from real Azure production traces: even with full continuous batching across many concurrent decode requests, the decode phase consistently underutilizes GPU compute. The SPAD research group at UT Austin further confirmed the hardware insensitivity from the opposite direction: reducing HBM bandwidth by 40 percent on the prefill chip increased prefill latency by only 17 percent, and reducing compute capacity by 50 percent on the decode chip increased decode latency by only 22 percent. The two phases are in fundamentally different hardware regimes with almost no overlap in their optimal hardware profiles.

---

## 2. The Interference Problem: What Happens When Both Phases Share a GPU

Given the structural asymmetry described above, the question becomes: what happens when you run both phases on the same pool of GPUs, as essentially every LLM serving system did before 2024?

### 2.1 The Mechanism of Interference

In a standard monolithic serving deployment, a single GPU pool processes incoming requests with continuous batching. At any given moment, the batch contains a mix of requests in prefill and requests in decode. This mixing is the source of the problem.

```
# Pseudocode: Monolithic continuous-batching scheduler
# This is the collocated architecture -- both phases on the same GPU

class MonolithicScheduler:
    def __init__(self):
        self.prefill_queue = []   # incoming requests waiting for prefill
        self.decode_queue  = []   # requests actively generating tokens

    def schedule_one_step(self):
        # Build the batch for this forward pass
        batch = []

        # First, admit new prefill requests (up to memory limit)
        while self.prefill_queue and gpu_memory_available():
            req = self.prefill_queue.pop(0)
            batch.append(PrefillItem(req))     # full prompt, all tokens

        # Also include all active decode requests
        for req in self.decode_queue:
            batch.append(DecodeItem(req))      # one new token per request

        # Run ONE forward pass for the entire mixed batch
        # *** This is where the interference happens ***
        # Prefill items are compute-bound (large matmuls over full seq)
        # Decode items are memory-bound (reading KV cache per token)
        # They compete for the same GPU for the entire step duration.
        outputs = gpu.forward(batch)

        # Process results
        for item, output in zip(batch, outputs):
            if isinstance(item, PrefillItem):
                self.decode_queue.append(item.req)   # prefill done → move to decode
            elif isinstance(item, DecodeItem):
                item.req.emit_token(output)
                if output == EOS:
                    self.decode_queue.remove(item.req)

    # Problem: when a long prefill runs (e.g. 4096 tokens → 300ms),
    # ALL decode requests in decode_queue wait the entire 300ms.
    # Their TPOT spikes. Users see their stream pause mid-sentence.
    # And when decode_queue is full, new prefill requests queue behind
    # them → TTFT rises even at moderate load.
```

When a new request arrives and its prefill begins, the GPU must allocate substantial compute time to processing the input sequence. A 4,096-token prompt requires hundreds of milliseconds of dense matrix computation. During this time, every decode request must wait. From the perspective of a user watching streaming output, the text stops mid-sentence. The prefill of someone else's long prompt has stalled their generation.

### 2.2 The Resource Coupling Problem

Beyond scheduling-level interference, there is a deeper architectural problem that scheduling cannot fix: **resource coupling**. The optimal tensor parallelism (TP) degree for the two phases is opposite. A lower TP degree means fewer all-reduce operations per layer, reducing per-request TTFT. A higher TP degree means each GPU holds a smaller shard, reducing the HBM each GPU reads per decode step. In a collocated pool, you must choose one TP value that is suboptimal for both phases.

```
# Pseudocode: The TP configuration dilemma in a collocated pool

# What each phase wants:
prefill_optimal_tp = 2   # fewer all-reduce calls → lower latency per request
decode_optimal_tp  = 8   # smaller shard per GPU → each GPU reads less HBM per step

# In a collocated pool, you must pick one value.
# Any choice is a compromise.
collocated_tp = 4        # neither optimal: TTFT higher than needed,
                         #                  ITL higher than needed

# In a disaggregated pool:
prefill_pool_tp = 2      # optimal for TTFT
decode_pool_tp  = 8      # optimal for ITL
# Both pools operate at their hardware optimum simultaneously.
```

### 2.3 Quantifying the Damage

DistServe (OSDI 2024) measured the interference effect directly. Collocated systems showed three to five times higher TPOT variance compared to disaggregated systems, even with the most sophisticated scheduling mitigation applied. Decode ITL spiked in direct positive correlation with prefill batch size. TTFT rose above SLO thresholds even at request rates well below the server's peak throughput capacity.

---

## 3. The Aggregation Baseline: Can Better Scheduling Solve the Problem?

Before accepting that the problem requires an architectural solution, it is worth asking whether improved scheduling can close the gap. SARATHI (OSDI 2024) represents the definitive attempt to answer this question.

### 3.1 Chunked Prefill: The Core Mechanism

The central idea of SARATHI is **chunked prefill**: instead of processing a large prefill in a single uninterrupted forward pass, split it into fixed-size chunks and interleave each chunk with decode steps. A 4,096-token prefill processed in chunks of 512 tokens becomes eight steps, each taking approximately 100 ms instead of one step taking 800 ms.

```
# Pseudocode: SARATHI chunked prefill scheduler

CHUNK_SIZE = 512   # tokens per prefill chunk (configurable)

class SARATHIScheduler:
    def __init__(self):
        self.prefill_queue = []    # each entry: (request, tokens_remaining, chunk_offset)
        self.decode_queue  = []

    def schedule_one_step(self):
        batch = []

        # Process ONE chunk of the oldest pending prefill
        if self.prefill_queue:
            req, remaining, offset = self.prefill_queue[0]
            chunk = req.tokens[offset : offset + CHUNK_SIZE]
            batch.append(PrefillChunkItem(req, chunk))

            if remaining <= CHUNK_SIZE:
                self.prefill_queue.pop(0)            # prefill complete
                self.decode_queue.append(req)
            else:
                self.prefill_queue[0] = (req, remaining - CHUNK_SIZE, offset + CHUNK_SIZE)

        # Fill remaining batch capacity with decode requests
        for req in self.decode_queue:
            batch.append(DecodeItem(req))             # piggyback decodes

        # Forward pass for the mixed batch
        # Max stall for decode = time to process ONE chunk (not full prefill)
        outputs = gpu.forward(batch)

        for item, output in zip(batch, outputs):
            if isinstance(item, DecodeItem):
                item.req.emit_token(output)

    # What SARATHI achieves:
    # - Max decode stall reduced from (full prefill duration) to (one chunk duration)
    # - Decode requests no longer wait 800ms; they wait at most ~100ms per chunk
    # - Measured: up to 10x decode throughput improvement (LLaMA-13B, A6000)

    # What SARATHI cannot fix:
    # - Residual stall per chunk still exists (even 1-token chunk = 1 GPU step)
    # - Resource coupling: same TP config serves both phases
    # - Hardware mismatch: idle tensor cores during decode still idle
```

### 3.2 The Hard Limits of Scheduling

SARATHI's measured results are substantial — up to 10× decode throughput improvement for LLaMA-13B. But its gains have a ceiling that no scheduling improvement can break through, because the ceiling is not a scheduling limit — it is a physics limit.

Even a one-token prefill chunk requires exclusive GPU access for a full forward step. At high prefill rates, the residual interference from chunk-level stalls remains significant. DistServe measured three to five times higher TPOT variance with chunked prefill compared to disaggregated serving. More fundamentally, chunked prefill does nothing about resource coupling — both phases still run on the same hardware with the same TP configuration.

It is worth noting that SARATHI and disaggregation are not mutually exclusive. In SGLang's disaggregated deployment, the prefill server uses chunked prefill internally to prevent any single large prompt from delaying the KV transfer handoff. Chunked prefill within a dedicated prefill pool is still useful; it just can no longer cause decode interference, because decode runs on entirely separate hardware.

---

## 4. The Correct Optimisation Target: Goodput

Before examining the disaggregated solution in detail, it is important to establish the correct metric by which to measure success. The conventional metric — raw throughput in requests per second — systematically obscures the interference problem and makes suboptimal solutions appear satisfactory.

### 4.1 Why Raw Throughput Misleads

A serving system can achieve high throughput while violating the latency constraints that users actually care about. Suppose a system processes 100 requests per second, but 30 percent experience TTFT or TPOT violations. From a user-experience perspective, those 30 percent are failures. Raw throughput counts them as successes.

### 4.2 Goodput: The Principled Alternative

DistServe introduced **goodput** as the correct optimisation target: the number of requests that satisfy both the TTFT SLO and the TPOT SLO per unit time.

```
# Pseudocode: Goodput measurement

function measure_goodput(requests, TTFT_SLO_ms, TPOT_SLO_ms, window_seconds):
    """
    A request counts toward goodput ONLY if it satisfies both SLOs simultaneously.
    """
    good_count = 0

    for req in requests:
        ttft_ok = req.actual_ttft_ms  <= TTFT_SLO_ms
        tpot_ok = req.actual_tpot_ms  <= TPOT_SLO_ms

        # BOTH conditions must hold -- one violation disqualifies the request
        if ttft_ok and tpot_ok:
            good_count += 1

    return good_count / window_seconds   # goodput = good requests per second


# Illustration: collocated vs disaggregated at the same request rate
requests_per_second  = 100

# Collocated (interference causes correlated violations):
# TTFT violations: 20%  TPOT violations: 25%  Both violated: 30%
collocated_good      = 70    # only 70% satisfy both SLOs
collocated_goodput   = 70.0  # goodput = 70 req/sec

# Disaggregated (interference eliminated):
# TTFT violations: 2%   TPOT violations: 1%   Both violated: 0.1%
disaggregated_good   = 97    # 97% satisfy both SLOs
disaggregated_goodput = 97.0 # goodput = 97 req/sec

# Raw throughput comparison: both show 100 req/sec -- no difference visible
# Goodput comparison: 97 vs 70 -- 1.39x difference visible
# Under tighter SLOs, this ratio grows to 7.4x (DistServe result)
```

DistServe's headline result — 7.4 times more requests per second at the same SLO constraints — is measured in goodput. Under raw throughput comparison, the improvement appears smaller, because the baseline's raw throughput counts requests that are failing SLOs. When measured correctly, with only SLO-satisfying requests counted, the advantage of eliminating interference is far more pronounced.

---

## 5. The Disaggregated Solution: Separating the Phases

With the problem established precisely and the limits of scheduling-based mitigation understood, the architectural solution becomes clear: the two phases must run on separate hardware that can be independently configured and independently scaled.

### 5.1 The Architecture

A disaggregated serving system replaces the monolithic GPU pool with three distinct components: a **router**, a **prefill pool**, and a **decode pool**. The router is the single client-facing entry point — it holds no model state and performs no inference computation. The prefill pool runs model forward passes over prompts and builds KV caches. The decode pool receives KV caches and runs autoregressive generation.

```
# Pseudocode: Request lifecycle in a disaggregated system

class DisaggregatedRouter:
    def __init__(self, prefill_workers, decode_workers):
        self.prefill_workers = prefill_workers   # list of PrefillWorker
        self.decode_workers  = decode_workers    # list of DecodeWorker

    def handle_request(self, request):
        # Step 1: select a prefill worker (round-robin or load-aware)
        prefill_worker = self.select_prefill_worker()

        # Step 2: prefill worker processes the prompt
        #         builds KV cache in its own VRAM
        #         returns pointer to KV pages + completion signal
        kv_metadata = prefill_worker.run_prefill(request.prompt)

        # Step 3: select a decode worker, transfer KV cache to it
        decode_worker = self.select_decode_worker()
        decode_worker.receive_kv(kv_metadata, src=prefill_worker)
        # KV cache moves over network (InfiniBand RDMA or NVLink)

        # Step 4: decode worker generates tokens, streams back to client
        for token in decode_worker.run_decode(kv_metadata):
            request.stream_token(token)
            if token == EOS:
                break


class PrefillWorker:
    """
    Configured for compute throughput:
      tp_size = 2 (fewer all-reduces → lower per-request TTFT)
      GPU type = H100 (highest FLOPS)
      batch strategy = pack similar-length prompts for FLOP efficiency
    """
    def run_prefill(self, prompt_tokens):
        kv_cache = forward_pass(prompt_tokens)    # builds KV for all layers
        return KVMetadata(pages=kv_cache, worker_id=self.id)


class DecodeWorker:
    """
    Configured for memory-bandwidth throughput:
      tp_size = 8 (smaller KV shard per GPU → lower HBM reads per step)
      GPU type = A100 (sufficient BW, lower cost)
      batch strategy = maximize concurrent requests (amortise HBM reads)
    """
    def receive_kv(self, kv_metadata, src):
        # Transfer KV from prefill worker VRAM to this worker's VRAM
        rdma_transfer(src.vram[kv_metadata.pages], self.vram)

    def run_decode(self, kv_metadata):
        while True:
            token = decode_one_step(kv_metadata)    # reads KV from HBM
            yield token
            if token == EOS: return


# What changes compared to the monolithic architecture:
# - Prefill and decode never occupy the same GPU simultaneously → no interference
# - Each pool has its own TP config optimized for its workload
# - TTFT SLO violated? Scale prefill_workers. TPOT violated? Scale decode_workers.
# - Prefill pool → H100. Decode pool → A100. Different hardware per phase.
```

### 5.2 What Separation Enables

The immediate benefit is elimination of scheduling interference. Because the decode pool never runs prefill, decode ITL is perfectly stable. Because the prefill pool never runs decode, prefill throughput is not limited by decode queue occupancy.

But the more significant benefit is the removal of resource coupling. The prefill pool uses a TP degree chosen to minimize TTFT; the decode pool uses a TP degree chosen to maximize HBM utilization. These are different numbers, and both pools operate at their hardware optimum simultaneously — something structurally impossible in a collocated system.

DistServe's evaluation showed 7.4× more goodput at the same SLO thresholds, and the ability to satisfy SLO constraints 12.6× tighter than the collocated baseline at the same request rate.

---

## 6. Hardware Heterogeneity: The Cost Consequence of Separation

Once the two phases are running on separate hardware, the two pools need not use the same GPU model. This insight has direct and quantifiable cost implications.

### 6.1 The Mismatch at the Silicon Level

An H100 SXM delivers 80 teraFLOPS FP16 and 3.35 TB/s HBM3 bandwidth. During decode, the arithmetic intensity is 1 FLOP/byte — the 80 teraFLOPS go almost entirely unused while memory bandwidth runs at saturation. An A100 80GB delivers 77 teraFLOPS and 2.0 TB/s HBM2e at approximately half the cost. For a decode-only worker, the A100 provides roughly 60 percent of the H100's decode throughput at 50 percent of the cost — a favorable ratio.

```
# Pseudocode: Per-dollar decode throughput comparison

def decode_throughput(hbm_bandwidth_TBps):
    """
    Decode throughput is limited by HBM bandwidth, not FLOPS.
    More bandwidth → more KV data delivered per second → more tokens/sec.
    """
    # ITL (inter-token latency) ∝ kv_cache_size / hbm_bandwidth
    # throughput = 1 / ITL ∝ hbm_bandwidth
    return hbm_bandwidth_TBps  # proportional relationship

A100_bw   = 2.0    # TB/s HBM2e
H100_bw   = 3.35   # TB/s HBM3
A100_cost = 15_000 # USD per GPU (approximate)
H100_cost = 30_000 # USD per GPU (approximate)

A100_decode_throughput_per_dollar = decode_throughput(A100_bw) / A100_cost
H100_decode_throughput_per_dollar = decode_throughput(H100_bw) / H100_cost

# A100: 2.0  / 15,000 = 1.33e-4
# H100: 3.35 / 30,000 = 1.12e-4
# A100 delivers ~19% more decode throughput per dollar than H100

# For prefill: throughput is limited by FLOPS, not HBM bandwidth
def prefill_throughput(tflops):
    return tflops   # proportional relationship

A100_prefill_per_dollar = prefill_throughput(77)  / A100_cost   # 5.1e-3
H100_prefill_per_dollar = prefill_throughput(989) / H100_cost   # 3.3e-2  (FP8)
# H100 is ~6x better per dollar for FP8 prefill (Blackwell-era models)

# Conclusion: use H100 for prefill pool, A100 for decode pool.
# Each dollar contributes more to the workload it is actually suited for.
```

### 6.2 The Splitwise Measurement

Splitwise formalized this reasoning and backed it with experimental data from real Azure service traces. At the same total cost budget, an H100-prefill + A100-decode cluster delivered 2.35× more throughput than an all-H100 cluster. At the same throughput target, the heterogeneous cluster cost 20 percent less. Perplexity AI confirmed the same principle in production, noting that disaggregation gives them operational freedom to use different GPU products for each phase based on each phase's actual hardware requirements.

---

## 7. The Transfer Problem: Moving the KV Cache

Disaggregation introduces a problem that does not exist in collocated serving: the KV cache, built by the prefill worker in its local VRAM, must be physically moved across a network to the decode worker's VRAM before token generation can begin. This transfer is the primary cost that disaggregation adds, and it determines whether the interference-elimination benefit is large enough to justify the architectural complexity.

### 7.1 The Size of the Cache

The KV cache size is determined by the model architecture and the prompt length.

```
# Pseudocode: KV cache size formula

def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes=2):
    """
    dtype_bytes = 2 for FP16, 1 for FP8, 4 for FP32
    Factor of 2: one Key tensor + one Value tensor per layer
    """
    per_token = n_layers * n_kv_heads * head_dim * 2 * dtype_bytes
    return per_token * seq_len


# LLaMA-3.1-70B (grouped-query attention, 8 KV heads), FP16
llama70b_per_token = kv_cache_bytes(
    n_layers   = 80,
    n_kv_heads = 8,
    head_dim   = 128,
    seq_len    = 1,        # bytes per token
    dtype_bytes= 2
)
# = 80 * 8 * 128 * 2 * 2 = 327,680 bytes/token ≈ 320 KB/token

llama70b_4k  = llama70b_per_token * 4_096   # ≈ 1.34 GB
llama70b_128k = llama70b_per_token * 131_072 # ≈ 40 GB

# DeepSeek-V3 uses Multi-head Latent Attention (MLA) -- compresses KV
# Smaller latent vectors → smaller cache per token than standard GQA
deepseek_v3_4k = kv_cache_bytes(n_layers=61, n_kv_heads=8, head_dim=128,
                                 seq_len=4096, dtype_bytes=2)  # ≈ 0.5 GB

# At 100 requests/second with 4K-token LLaMA-70B prompts:
# Transfer requirement = 1.34 GB/req * 100 req/sec = 134 GB/sec
# This must fit entirely within the network bandwidth between pools.
```

### 7.2 The Network Speed Requirement and the GPUDirect Bypass

For a TTFT budget of 500 ms with 200 ms prefill time, the remaining 300 ms must cover the KV transfer. With a 1.34 GB payload, that requires at minimum 4.5 GB/s of effective network bandwidth.

The word "effective" is critical. Without GPUDirect RDMA, the transfer must cross PCIe twice — once from the prefill GPU into CPU RAM, and once from the decode server's CPU RAM into the decode GPU. Each crossing halves the effective bandwidth through PCIe bus sharing and CPU involvement.

```
# Pseudocode: Two transfer paths -- CPU-mediated vs GPUDirect RDMA

# PATH A: CPU-mediated (no GPUDirect) -- the naive path
function transfer_kv_cpu_mediated(kv_tensor, prefill_gpu, decode_gpu_remote):
    # Step 1: GPU → CPU RAM via PCIe  (PCIe Gen5 x16 ≈ 64 GB/s theoretical)
    cpu_buffer = prefill_gpu.to_host(kv_tensor)      # ~500ms for 1.34GB at real rates
    # PCIe is SHARED with other devices (NVMe, NICs, other GPUs)
    # Effective rate: ~6 GB/s

    # Step 2: CPU RAM → NIC (DMA from pinned memory)
    nic.send(cpu_buffer, dst=decode_gpu_remote.host)  # 100 GbE → 12.5 GB/s wire rate

    # Step 3 (remote): NIC → CPU RAM (DMA into pinned memory)
    remote_cpu_buffer = decode_gpu_remote.host.recv()

    # Step 4 (remote): CPU RAM → GPU VRAM via PCIe
    decode_gpu_remote.from_host(remote_cpu_buffer)

    # Total effective rate: bottlenecked by PCIe sharing ≈ 6 GB/s
    # Transfer time for 1.34 GB ≈ 220 ms   (73% of a 300ms budget)


# PATH B: GPUDirect RDMA -- the production path
function transfer_kv_gpudirect(kv_tensor, prefill_gpu, decode_gpu_remote):
    # RDMA NIC reads directly from prefill GPU VRAM (zero CPU involvement)
    # NIC is on the SAME PCIe root complex as the GPU → full NIC bandwidth
    # Data flows: GPU VRAM → NIC → InfiniBand → remote NIC → remote GPU VRAM

    # Register GPU memory with RDMA subsystem (done once at startup)
    prefill_rdma_key = rdma.register(prefill_gpu.vram)
    decode_rdma_key  = rdma.register(decode_gpu_remote.vram)

    # Single RDMA write: prefill GPU VRAM → decode GPU VRAM
    # No CPU involvement; no PCIe double-crossing
    rdma.write(
        src      = kv_tensor,
        src_key  = prefill_rdma_key,
        dst      = decode_gpu_remote.vram.allocate(kv_tensor.size),
        dst_key  = decode_rdma_key,
        protocol = "InfiniBand_NDR"   # 400 Gbps = 50 GB/s effective
    )
    # Transfer time for 1.34 GB ≈ 27 ms   (9% of a 300ms budget)
    # 8x faster than the CPU-mediated path


# Comparison:
# CPU-mediated (100 GbE):         220 ms  → uses 73% of TTFT budget
# GPUDirect RDMA (IB NDR):         27 ms  → uses 9% of TTFT budget
# GPUDirect RDMA (IB HDR):         54 ms  → uses 18% of TTFT budget
# NVLink (same server, NVLink 4): ~1.5 ms → essentially free
```

GPUDirect RDMA is not optional for production disaggregation. It is the technology that makes the transfer cost acceptable. This is why the entire software ecosystem for KV transfer — Mooncake, NIXL, and the vLLM connector abstractions — is built on top of it.

---

## 8. The Transfer Layer: Three Approaches

The transfer problem has two dimensions. The first is the physical transport dimension: given GPUDirect RDMA as the foundation, how do you use available network hardware as efficiently as possible? The second is the abstraction dimension: how do you expose KV transfer as a reusable interface that inference frameworks can use without coupling themselves to a specific transport technology? The ecosystem has developed three complementary answers.

### 8.1 The Production Transfer Engine: Mooncake

Mooncake (FAST 2025) is the KV transfer engine built by Moonshot AI for the Kimi service. It represents a production-system approach: start with the physical reality of inference server hardware and solve every obstacle to maximum transfer throughput.

The first obstacle is NIC-GPU path non-uniformity. Modern inference servers have multiple CPU sockets, NUMA nodes, PCIe root complexes, GPUs, and RDMA NICs. A NIC on the same PCIe root complex as the source GPU provides full bandwidth; a NIC behind a NUMA interconnect provides 50 percent or less.

```
# Pseudocode: Mooncake topology-aware NIC selection

# At server startup: build a topology matrix
function build_topology_matrix(gpus, nics):
    """
    Maps each (GPU, NIC) pair to its effective bandwidth.
    Measured from the PCIe/NUMA topology of this specific server.
    """
    topology = {}
    for gpu in gpus:
        for nic in nics:
            # Check if NIC and GPU share the same PCIe root complex
            if same_pcie_root_complex(gpu, nic):
                bandwidth = nic.max_bandwidth     # e.g. 50 GB/s for NDR
            elif same_numa_node(gpu, nic):
                bandwidth = nic.max_bandwidth * 0.7   # NUMA hop penalty
            else:
                bandwidth = nic.max_bandwidth * 0.4   # UPI/QPI crossover penalty
            topology[(gpu.id, nic.id)] = bandwidth
    return topology


# At transfer time: multi-NIC pooling for large payloads
function transfer_kv_mooncake(kv_tensor, src_gpu, topology, available_nics):
    """
    Slice the transfer across multiple NICs simultaneously.
    Each slice uses the NIC with the best affinity for the source GPU.
    """
    # Rank NICs by affinity to src_gpu
    ranked_nics = sorted(
        available_nics,
        key=lambda nic: topology[(src_gpu.id, nic.id)],
        reverse=True
    )

    # Slice tensor across top-N NICs
    slices = split_tensor(kv_tensor, n=len(ranked_nics))

    # Submit all slices in parallel (non-blocking)
    handles = []
    for slice_data, nic in zip(slices, ranked_nics):
        handle = nic.rdma_write_async(slice_data, dst=decode_gpu_remote)
        handles.append(handle)

    # Wait for all slices to complete
    wait_all(handles)

    # For a server with 8 NICs at 50 GB/s each:
    # aggregate bandwidth ≈ 400 GB/s
    # 40 GB KV cache (LLaMA-70B, 128K context): ~100ms instead of ~800ms


# Transport backend selection (same API, different physical path)
function select_backend(src_gpu, dst_gpu, topology_info):
    if src_gpu.server == dst_gpu.server:
        return NVLinkBackend()           # intra-node: fastest
    elif same_nvl72_rack(src_gpu, dst_gpu):
        return MNNVLBackend()            # intra-rack NVLink: ~1000 GB/s
    else:
        return InfiniBandRDMABackend()   # inter-node: 50-100 GB/s
```

In production on Kimi, Mooncake's optimizations produced 75 percent more requests handled under SLO constraints and up to 525 percent throughput increase for long-context workloads. Kimi K2 (128 H200 GPUs, July 2025) achieved 224,000 tokens per second of prefill throughput and 288,000 tokens per second of decode throughput.

### 8.2 The Vendor-Agnostic Library: NIXL

NIXL (NVIDIA Inference Xfer Library) takes a different design goal: work with any interconnect, expose one unified API, and let the framework not care about the underlying transport.

```
# Pseudocode: NIXL Transfer Agent -- unified interface across all memory types

class NIXLTransferAgent:
    """
    Each server process runs one Transfer Agent.
    It manages memory registration, backend selection, and async transfers.
    """
    def __init__(self):
        # Register all accessible memory regions at startup
        self.gpu_vram  = rdma.register(cuda.get_vram())      # GPUDirect RDMA
        self.cpu_dram  = rdma.register(pin_memory(cpu_ram))  # CPU pinned
        self.nvme      = gds.register(local_nvme())          # GPU Direct Storage
        # Caller uses the same API regardless of which memory holds the data

        # Exchange registration metadata with remote agents (cached in etcd)
        self.remote_keys = etcd.get_or_create(f"nixl/keys/{self.node_id}")


    # ASYNC TRANSFER -- the key capability distinguishing NIXL from raw RDMA
    def submit_write(self, src_buffer, dst_agent, dst_buffer):
        """
        Non-blocking: submit and return immediately with a handle.
        The transfer runs in the background.
        """
        handle = self._rdma_write_async(
            src=src_buffer,
            dst=dst_agent.endpoint,
            dst_key=self.remote_keys[dst_agent.id],
            size=src_buffer.size
        )
        return handle   # caller can do other work while transfer runs

    def check_complete(self, handle):
        return handle.is_done()

    def wait(self, handle):
        handle.block_until_done()


# How the decode worker exploits NIXL's async API:
# Layers arrive one at a time as prefill worker completes them.

function decode_with_overlap(kv_metadata, nixl_agent):
    """
    Overlap KV transfer and decode computation.
    Begin decoding layers as soon as they arrive -- don't wait for all layers.
    """
    handles = []

    # Submit transfer for all layers non-blocking
    for layer_id in range(n_layers):
        src_buffer = prefill_kv[layer_id]
        dst_buffer = decode_kv_buffer[layer_id]
        handle = nixl_agent.submit_write(src_buffer, remote_agent, dst_buffer)
        handles.append((layer_id, handle))

    # Process each layer as soon as its transfer completes
    for layer_id, handle in handles:
        nixl_agent.wait(handle)          # block only until this layer is ready
        decode_attention_layer(layer_id) # compute attention using received KV

    # Effective transfer cost < raw bandwidth calculation
    # because compute and transfer run concurrently for most layers.
```

NIXL serves three distinct use cases within the same interface: KV transfer between disaggregated workers, loading KV caches from persistent storage for long-context requests, and moving expert activations for MoE all-to-all communication. All three use the same Transfer Agent and the same submit/check API.

### 8.3 The Framework Abstraction: vLLM's Connector Interface

While Mooncake and NIXL solve the transport problem at the systems level, inference frameworks need a stable API surface that lets them express "this worker produces KV caches" or "this worker consumes KV caches" without coupling to a specific transport. vLLM's connector architecture provides this abstraction.

```
# Pseudocode: vLLM BaseKVConnector -- the framework-level interface

class BaseKVConnector(ABC):
    """
    Abstract interface that any KV transfer mechanism must implement.
    The inference engine calls these two methods; it does not know
    which transport (RDMA, NCCL, NVLink) is underneath.
    """

    @abstractmethod
    def send_kv_caches_and_hidden_states(
        self, model_input, kv_caches, hidden_states
    ) -> None:
        """
        Called on the PREFILL instance after its forward pass.
        Writes KV tensors to the transfer buffer.
        The connector handles the actual transport (RDMA, NCCL, etc.)
        """
        ...

    @abstractmethod
    def recv_kv_caches_and_hidden_states(
        self, model_input, kv_caches
    ) -> tuple[Tensor, bool]:
        """
        Called on the DECODE instance before its forward pass.
        Returns: (hidden_states, bypass_model_exec)

        bypass_model_exec = True  → skip decode's own prefill computation
                                    KV cache already arrived via transfer
        bypass_model_exec = False → something went wrong; run prefill locally
        """
        ...


class KVLookupBufferBase(ABC):
    """
    Governs buffer ownership between prefill and decode instances.
    The 'drop' in drop_select is critical: it ensures atomic ownership transfer.
    """

    @abstractmethod
    def insert(self, input_tokens, key, value, hidden):
        """Called by prefill instance: place KV into the shared buffer."""
        ...

    @abstractmethod
    def drop_select(self, input_tokens):
        """
        Called by decode instance: ATOMICALLY retrieve and REMOVE the KV entry.
        'drop' prevents double consumption: once taken, no other decode
        instance can claim the same KV cache.
        Returns: (key, value, hidden) or None if not yet available.
        """
        ...


# Six concrete connectors -- same interface, different transports:
# NixlConnector        → RDMA via UCX    (production, InfiniBand/RoCEv2)
# MooncakeConnector    → RDMA + topology-aware multi-NIC
# P2pNcclConnector     → NCCL P2P        (no RDMA NICs, uses NVLink/PCIe)
# LMCacheConnectorV1   → NIXL + storage  (persistent KV + cross-engine sharing)
# MultiConnector       → chains any combination (RDMA primary, NCCL fallback)
# ExampleConnector     → reference implementation for custom connectors

# The vLLM inference engine calls the connector without knowing which is active:
connector = NixlConnector(config)   # or MooncakeConnector, P2pNcclConnector, ...

if worker_role == "prefill":
    result = model.forward(input)
    connector.send_kv_caches_and_hidden_states(input, result.kv, result.hidden)

elif worker_role == "decode":
    kv, bypass = connector.recv_kv_caches_and_hidden_states(input, local_kv)
    if bypass:
        tokens = model.decode_only(kv)    # skip prefill, use received KV
    else:
        tokens = model.full_forward(input) # fallback: run locally
```

The vLLM documentation states this plainly: "Disaggregated prefill does not improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." This statement correctly identifies what disaggregation is for.

---

## 9. The Orchestration Problem: What Transfer Alone Cannot Solve

Once the transfer layer is in place, a new class of problems emerges at production scale. These are not transport problems — they are **coordination problems**: which worker is available, which one has the right KV prefix cached, and how should pool sizes adjust as traffic shifts?

### 9.1 The Four Planes of Production Orchestration

NVIDIA Dynamo (GTC 2025) is the clearest public statement of what production orchestration requires. Its central claim: managing disaggregated serving requires four separate communication planes, each handling a different concern.

```
# Pseudocode: Dynamo's four-plane architecture

# PLANE 1: Request Plane (data path -- every system has this)
class RequestPlane:
    def route(self, request):
        prefill_worker = self.discovery.get_available_prefill()
        kv = prefill_worker.run_prefill(request)
        decode_worker = self.discovery.get_available_decode(kv.prefix_hash)
        return decode_worker.run_decode(kv)


# PLANE 2: Discovery Plane (dynamic worker registration via etcd)
class DiscoveryPlane:
    TTL_SECONDS = 10   # worker lease TTL

    def worker_startup(self, worker):
        # Worker publishes its record with a TTL lease
        record = {
            "endpoint":  worker.url,
            "role":      worker.role,       # "prefill" or "decode"
            "model":     worker.model_id,
            "load":      worker.queue_depth
        }
        etcd.put_with_lease(f"workers/{worker.id}", record, ttl=self.TTL_SECONDS)
        # Worker refreshes lease every TTL/3 seconds while alive

    def on_worker_death(self, worker_id):
        # Lease expires automatically after TTL_SECONDS
        # Router is notified via etcd watch → removes worker from routing table
        pass  # no manual intervention needed

    def get_available_prefill(self):
        # Router reads live worker list from etcd (updated continuously)
        return [w for w in etcd.list("workers/") if w.role == "prefill"]


# PLANE 3: Event Plane (async state propagation via pub/sub)
class EventPlane:
    def on_kv_cache_update(self, worker_id, new_prefixes):
        # Prefill workers publish their KV cache prefix state as it changes
        event_bus.publish(f"kv_state/{worker_id}", new_prefixes)

    def router_subscribe(self, router):
        # Router maintains a live routing table without polling
        event_bus.subscribe("kv_state/*", callback=router.update_routing_table)

    # Without the event plane: router must poll all 50+ workers every N seconds
    # Polling traffic at scale exceeds the actual inference request traffic.
    # Push-based events eliminate this overhead.


# PLANE 4: Control Plane / Planner (dynamic pool sizing via SLO feedback)
class Planner:
    TTFT_SLO      = 2000  # ms
    TPOT_SLO      = 100   # ms
    REACTION_TIME = 30    # seconds: min time between scaling actions
    COOLDOWN      = 120   # seconds: min time after scale-up before scale-down

    def monitor_loop(self):
        while True:
            ttft_p95 = metrics.get_ttft_p95(window_seconds=60)
            tpot_p95 = metrics.get_tpot_p95(window_seconds=60)

            if ttft_p95 > self.TTFT_SLO * 1.05:   # TTFT SLO violated
                self.scale_up("prefill")            # add prefill workers
            elif tpot_p95 > self.TPOT_SLO * 1.05: # TPOT SLO violated
                self.scale_up("decode")             # add decode workers

            sleep(self.REACTION_TIME)
```

### 9.2 KV-Aware Routing

Beyond the four planes, Dynamo's router implements a capability that provides significant TTFT reduction at no additional hardware cost: routing each request to the prefill worker that already has the most KV cached for that request's prefix.

```
# Pseudocode: KV-aware routing -- route to maximize prefix cache hits

class KVAwareRouter:
    def __init__(self, event_plane):
        # Maintain a live map: worker_id → set of cached token prefix hashes
        self.prefix_cache_state = {}
        event_plane.subscribe("kv_state/*", self.update_prefix_cache)

    def update_prefix_cache(self, worker_id, prefixes):
        self.prefix_cache_state[worker_id] = set(prefixes)

    def select_prefill_worker(self, request):
        req_prefix_hash = hash(request.prompt_tokens[:shared_prefix_len])

        # Find the worker with the longest matching cached prefix
        best_worker = None
        best_overlap = 0
        for worker_id, cached_prefixes in self.prefix_cache_state.items():
            overlap = len(cached_prefixes & {req_prefix_hash})
            if overlap > best_overlap:
                best_overlap = overlap
                best_worker  = worker_id

        if best_worker and best_overlap > CACHE_HIT_THRESHOLD:
            # Route to worker that already has KV for the shared context
            # → skips recomputing the shared portion → lower TTFT
            return best_worker
        else:
            # No useful cache hit; fall back to load-aware selection
            return self.select_least_loaded_worker()

# SGLang currently uses round-robin (equivalent to best_overlap = 0 always).
# Prefix-aware routing is planned; KV-aware routing is implemented in Dynamo.
```

---

## 10. When Disaggregation Is Not the Answer

Every preceding section has argued for disaggregation as the superior architecture. But this conclusion is conditional: disaggregation is superior when the binding SLO constraint is TPOT. When the binding constraint is TTFT, the situation is more nuanced.

### 10.1 The SLO Regime Framework

TaiChi (arXiv, August 2025) provides the most rigorous analysis of when disaggregation outperforms aggregation and when it does not. The core insight is that the two architectures occupy different regions of the SLO feasibility space.

```
# Pseudocode: The SLO regime analysis

def can_satisfy_slos(architecture, n_total_gpus, request_rate,
                     TTFT_SLO_ms, TPOT_SLO_ms):
    """
    Determine whether an architecture can satisfy both SLOs
    at the given request rate, for a fixed total GPU count.
    """
    if architecture == "aggregated":
        # All GPUs contribute to prefill: low TTFT possible
        # But decode is interrupted by prefill: TPOT variance is high
        effective_prefill_gpus = n_total_gpus
        ttft_achievable  = compute_ttft(effective_prefill_gpus, request_rate)
        tpot_achievable  = compute_tpot_with_interference(n_total_gpus, request_rate)

    elif architecture == "disaggregated":
        # Only prefill pool handles new requests
        prefill_gpus = n_total_gpus * prefill_fraction   # e.g. 25%
        decode_gpus  = n_total_gpus * (1 - prefill_fraction)  # 75%
        ttft_achievable = compute_ttft(prefill_gpus, request_rate)
        tpot_achievable = compute_tpot_no_interference(decode_gpus, request_rate)

    ttft_ok = ttft_achievable <= TTFT_SLO_ms
    tpot_ok = tpot_achievable <= TPOT_SLO_ms
    return ttft_ok and tpot_ok


# Result across SLO regimes (16 GPUs total, 0.25 prefill fraction):
#
# Tight TTFT (500ms) + Relaxed TPOT (500ms):
#   aggregated:   ttft=400ms ✓, tpot=300ms ✓  → can satisfy both
#   disaggregated: ttft=700ms ✗ (only 4 GPUs for prefill) → FAILS
#
# Relaxed TTFT (2000ms) + Tight TPOT (50ms):
#   aggregated:   ttft=400ms ✓, tpot=180ms ✗ (interference)  → FAILS
#   disaggregated: ttft=700ms ✓, tpot=40ms ✓                  → passes
#
# Tight TTFT (500ms) + Tight TPOT (50ms):
#   aggregated:   tpot=180ms ✗  → FAILS
#   disaggregated: ttft=700ms ✗ → FAILS
#   → neither pure architecture works; need TaiChi hybrid
```

### 10.2 TaiChi's Solution: Dynamic Worker Reassignment

TaiChi's response is a dynamic architecture that continuously reassigns workers between P-heavy and D-heavy roles based on live SLO measurements.

```
# Pseudocode: TaiChi SLO-driven worker reassignment loop

class TaiChiOrchestrator:
    def __init__(self, workers):
        self.p_heavy = set(workers[:len(workers)//4])  # initially 25% prefill
        self.d_heavy = set(workers[len(workers)//4:])  # initially 75% decode

    def control_loop(self):
        while True:
            ttft_violation_rate = metrics.ttft_violation_rate(last_60s)
            tpot_violation_rate = metrics.tpot_violation_rate(last_60s)

            if ttft_violation_rate > 0.05:   # >5% TTFT violations
                # Not enough prefill capacity; convert one D-heavy to P-heavy
                worker = pick_least_loaded(self.d_heavy)
                self.d_heavy.remove(worker)
                self.p_heavy.add(worker)
                worker.set_mode("p_heavy")   # drain decode queue, accept prefill
                print(f"TTFT pressure: moved {worker} to P-heavy "
                      f"({len(self.p_heavy)} P / {len(self.d_heavy)} D)")

            elif tpot_violation_rate > 0.05: # >5% TPOT violations
                # Not enough decode capacity; convert one P-heavy to D-heavy
                worker = pick_least_loaded(self.p_heavy)
                self.p_heavy.remove(worker)
                self.d_heavy.add(worker)
                worker.set_mode("d_heavy")
                print(f"TPOT pressure: moved {worker} to D-heavy "
                      f"({len(self.p_heavy)} P / {len(self.d_heavy)} D)")

            sleep(30)   # reaction time: don't over-react to transient spikes

    # At equilibrium: minimum P-heavy to satisfy TTFT SLO,
    #                 minimum D-heavy to satisfy TPOT SLO,
    #                 remainder unassigned or load-balanced.
    # Measured improvement: up to 77% goodput improvement over SOTA.
    # TTFT reduction vs pure disaggregation: up to 13.2x.
    # TPOT reduction vs pure aggregation:    up to 1.69x.
```

There is one category of workload where disaggregation is mandatory regardless of SLO regime: Mixture-of-Experts models like DeepSeek-V3. These models use expert parallelism libraries (DeepEP) that implement different communication dispatch modes for prefill and decode — high-throughput "normal" mode for prefill's large batches, and low-latency mode for decode's single-token batches. These modes are mutually exclusive on the same worker. Only by running prefill and decode on separate workers can each phase use its optimal dispatch mode.

---

## 11. The Handshake: How SGLang Implements the Transfer Correctly

The preceding sections have described the architecture and the transfer mechanisms. There is one implementation detail that determines whether the transfer is correct — the **pre-allocation handshake** that prevents the race condition inherent in KV transfer.

```
# Pseudocode: SGLang's pre-allocation handshake protocol
#
# The naive race condition:
#   Prefill finishes → tries to RDMA-write KV to decode worker
#   Decode worker has NOT pre-allocated space → write fails or overwrites
#   Fix: decode allocates FIRST, then prefill writes into pre-allocated space

class SGLangPrefillWorker:
    def handle_prefill_request(self, request):
        # Step 1: Negotiate with decode worker BEFORE running forward pass
        decode_worker = router.select_decode_worker(request)

        # Step 2: Decode worker pre-allocates KV cache pages in its VRAM
        #         and sends back the page indices
        kv_page_indices = decode_worker.preallocate_kv_pages(
            n_tokens=len(request.prompt_tokens)
        )
        # kv_page_indices tells us exactly WHERE in decode VRAM to write

        # Step 3: Run prefill forward pass, writing KV into PRE-ALLOCATED indices
        self.model.forward(
            request.prompt_tokens,
            kv_dest = decode_worker.vram[kv_page_indices]  # write directly
        )

        # Step 4: Notify decode worker that KV is ready
        decode_worker.notify_kv_ready(request.id)


class SGLangDecodeWorker:
    def preallocate_kv_pages(self, n_tokens):
        # Allocate KV cache space in advance
        # Returns page indices to the prefill worker
        pages = self.kv_allocator.allocate(n_tokens)
        return pages

    def notify_kv_ready(self, request_id):
        # Called by prefill worker after RDMA write is complete
        # Now it is safe to begin decoding
        self.decode_queue.add(request_id)
        # Autoregressive generation begins immediately


# Environment variables that control transfer behavior (SGLang):
# SGLANG_DISAGGREGATION_QUEUE_SIZE = 4   (parallel transfers to decode workers)
# SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = 300  (seconds to wait for preallocate)
# SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = 5.0 (health check interval)
# SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK  (use NVLink for intra-rack NVL72)
```

---

## 12. The Decision Framework: When and How to Disaggregate

The preceding sections have developed the complete conceptual framework. This final section synthesizes the key decision criteria.

### 12.1 The Five Checks

```
# Pseudocode: The five-check decision function

def should_disaggregate(workload_profile, cluster_profile):
    """
    Returns: "disaggregate", "aggregate", or "hybrid"
    """

    # CHECK 1: Decode/prefill time ratio
    decode_fraction = workload_profile.avg_decode_time / workload_profile.avg_total_time
    if decode_fraction < 0.70:
        note("Low decode fraction -- interference-elimination benefit is moderate")

    # CHECK 2: SLO regime (TaiChi's framework)
    ttft_tight = workload_profile.ttft_slo_ms < 500
    tpot_tight = workload_profile.tpot_slo_ms < 100

    if ttft_tight and not tpot_tight:
        return "aggregate"      # disaggregation hurts TTFT; aggregation is better
    elif tpot_tight and not ttft_tight:
        pass                    # disaggregation is the right choice; continue checks
    elif ttft_tight and tpot_tight:
        return "hybrid"         # TaiChi-style dynamic switching needed

    # CHECK 3: Transfer cost viability
    kv_size_gb = compute_kv_cache_size(
        workload_profile.median_prompt_tokens,
        cluster_profile.model
    )
    available_bw_GBps = cluster_profile.rdma_bandwidth_GBps
    transfer_time_ms  = (kv_size_gb / available_bw_GBps) * 1000

    ttft_budget_ms       = workload_profile.ttft_slo_ms
    prefill_compute_ms   = workload_profile.avg_prefill_time_ms
    transfer_budget_ms   = ttft_budget_ms - prefill_compute_ms

    if transfer_time_ms > transfer_budget_ms * 0.50:
        # Transfer consumes more than 50% of remaining TTFT budget
        if not cluster_profile.has_rdma:
            return "aggregate"  # CPU-mediated transfer is too slow; disagg not viable
        note(f"Transfer ({transfer_time_ms:.0f}ms) is tight; verify NIC affinity")

    # CHECK 4: Cluster scale
    if cluster_profile.total_gpus < 16:
        note("Small cluster -- overhead may exceed utilisation gain; evaluate carefully")
        if cluster_profile.total_gpus < 8:
            return "aggregate"

    # CHECK 5: Prefix cache hit rate
    if workload_profile.prefix_cache_hit_rate > 0.80:
        note("High prefix cache hit rate -- decode workers already hold most KV. "
             "Consider hybrid: local prefill for cache hits, remote for cold prompts")

    # All checks passed: disaggregation is the right choice
    return "disaggregate"


# Cost estimate if disaggregating with hardware heterogeneity:
def cost_reduction_estimate(current_cluster_cost):
    interference_elimination_gain = 1.25   # ~25% more goodput at same cost
    heterogeneous_hw_savings       = 0.80   # ~20% lower cluster cost (Splitwise)
    combined = interference_elimination_gain / heterogeneous_hw_savings
    # ≈ 2.35x more throughput at same cost, or 20% cost reduction at same throughput
    return current_cluster_cost * heterogeneous_hw_savings
```

### 12.2 The Cost Arithmetic

Disaggregation reduces per-token serving cost through two independent mechanisms. Interference elimination means more requests complete within SLO bounds, so effective goodput increases at the same hardware budget, reducing cost per correctly-served request. Hardware heterogeneity means the decode pool runs on less expensive hardware without quality loss, directly reducing the per-GPU cost of the cluster's largest pool. Splitwise's measurements suggest 20 percent cost reduction at equivalent throughput as the combined effect of these two mechanisms. Infrastructure analysts have estimated 15 to 40 percent total cluster cost reduction at production scale.

The eighteen-month adoption arc from first publication to broad production deployment is, in itself, a form of evidence. Engineering organizations do not adopt architectural complexity without a compelling return. Perplexity, Meta, LinkedIn, Mistral, Moonshot AI, HuggingFace, and the SGLang team all independently reached the same conclusion — and they did so because the cost arithmetic works in production, not just in benchmarks.
