# SGLang Dev Setup

## Environment

- `.conda/` at repo root — Python 3.11.15, CUDA toolkit 13.1 (pre-installed)
- SGLang repo: `REPOS/sglang/` (editable install)

## Install Steps

```bash
# 1. Upgrade pip
.conda/bin/pip install --upgrade pip setuptools wheel

# 2. PyTorch cu128
.conda/bin/pip install torch==2.9.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 3. SGLang editable + FlashInfer
cd REPOS/sglang/python
.conda/bin/pip install -e "." \
  --extra-index-url https://flashinfer.ai/whl/flashinfer-cubin/ \
  --extra-index-url https://flashinfer.ai/whl/flashinfer-python/

# 4. FlashInfer JIT cache (must match flashinfer version exactly — 0.6.7.post3)
.conda/bin/pip install "flashinfer-jit-cache==0.6.7.post3" \
  --extra-index-url https://flashinfer.ai/whl/cu128/

# 5. sgl-kernel
.conda/bin/pip install sglang-kernel
```

## Verify

```bash
.conda/bin/python -m sglang.check_env
```

Expected: `sglang`, `flashinfer_python`, `flashinfer_cubin`, `flashinfer_jit_cache` all at `0.6.7.post3`, PyTorch `2.9.1+cu128`.

## Run Server

```bash
./scripts/serve_qwen3.sh
# defaults: Qwen/Qwen3-0.6B-Instruct, port 30000, flashinfer backend
```

## Test Inference

```bash
./scripts/test_inference.sh
.conda/bin/python scripts/test_inference.py --stream
```

## Key Version Pins

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.9.1+cu128 |
| flashinfer_python | 0.6.7.post3 |
| flashinfer_cubin | 0.6.7.post3 |
| flashinfer_jit_cache | 0.6.7.post3 |
| sglang-kernel | 0.4.1 |
