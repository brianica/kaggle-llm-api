# Kaggle / Colab GPU LLM API

Turn a Kaggle or Google Colab GPU session into a public, password-protected,
**OpenAI-compatible** API for coding agents such as Aider, OpenCode, Cline, Roo Code,
Continue and the Codex CLI.

```
agent(s) ──HTTPS──▶ ngrok / Cloudflare tunnel ──▶ gateway :8000 (API key, least-busy routing, streaming)
                                                      ├──▶ vLLM :8001 (GPU 0, or GPU 0+1 split)
                                                      └──▶ vLLM :8002 (GPU 1, replica mode)
                                    or one llama.cpp server :8001 (layers split across both GPUs)
```

## Built for many connections

- **vLLM continuous batching**: many requests share each GPU pass instead of queueing.
- **Prefix caching**: the long prompt agents resend every turn is computed once.
- **Chunked prefill**: one huge prompt doesn't stall other users' streams.
- **Replica mode**: one engine per GPU, with requests sent to the least-busy one and failover if one is down.
- **Async gateway**: API-key auth, unlimited keep-alive connections, real-time token streaming.

## Quick start

The notebook detects whether it's on Kaggle or Colab, checks the GPU, and picks a model that fits (`PRESET = "auto"`).

**Secrets** (all optional; Kaggle: *Add-ons → Secrets*; Colab: the 🔑 icon in the left sidebar, with "Notebook access" on):

| Secret | Purpose |
|---|---|
| `API_KEY` | Key clients must send. Generated and printed if missing. |
| `NGROK_TOKEN` | ngrok tunnel (recommended; streams properly). `NGROK_DOMAIN` optionally sets your static domain. |
| `CF_TUNNEL_TOKEN` | Named Cloudflare tunnel pointing to `http://localhost:8000`. Set `CF_HOSTNAME` too. |
| `HF_TOKEN` | Only for gated models. |

With no tunnel token, it falls back to a free Cloudflare quick tunnel, which is meant for testing and may not stream.
After **Run All**, the notebook prints the base URL, API key, model name and copy-paste config for each coding tool.

### Kaggle

1. *Create → New Notebook → File → Import Notebook*, and upload `kaggle_gpu_llm_api.ipynb` or paste its GitHub URL.
2. *Settings*: Accelerator **GPU T4 ×2**, Internet **On**. (A P100 only works with the llama.cpp backend; vLLM needs a T4 or newer.)
3. **Run All**. To serve without a browser tab open, use **Save Version → Save & Run All (Commit)**, which runs for up to 12 hours.

### Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brianica/kaggle-llm-api/blob/main/kaggle_gpu_llm_api.ipynb)

1. Open the notebook with the badge above.
2. *Runtime → Change runtime type*: **T4 GPU** (free), or **L4** / **A100** (paid plans).
3. *Runtime → Run all*, and keep the tab open: Colab disconnects idle sessions.

| GPU | Auto preset | What to expect |
|---|---|---|
| Kaggle T4 ×2 | `qwen2.5-coder-14b`, split across GPUs | Several agents at once |
| Colab T4 (free) | `qwen2.5-coder-7b` | A handful of agents at once |
| Colab L4 (24 GB) | `qwen3-coder-30b` on vLLM, 32k context | Best agent model; a few agents at once |
| Colab A100 (40/80 GB) | `qwen3-coder-30b` on vLLM, 64k context | Many agents at once |

> **Colab terms:** Colab's FAQ restricts some activities on the free tier, including running web services unrelated to
> interactive notebook use and connecting through remote proxies. Serving an API through a tunnel may fall under that,
> and Colab can end the runtime or limit the account. Paid plans have fewer restrictions. Check the current FAQ first.

## Model presets

| Preset | Weights | Engine | Notes |
|---|---|---|---|
| `qwen3-coder-30b` | ~17-19 GB | vLLM, falling back to llama.cpp | **Best for agents**: MoE with 3B active parameters, trained for tool use. Needs ~20 GB |
| `qwen2.5-coder-14b` | ~10 GB | split across GPUs | Proven on T4 |
| `qwen2.5-coder-7b` | ~5.5 GB | one replica per GPU | Most simultaneous users |
| `qwen2.5-coder-32b` | ~19 GB | split across GPUs | Smartest; 16k context |
| `qwen3-8b` | ~6 GB | one replica per GPU | General + coding |

Or set a preset name yourself, or `PRESET = "custom"` with any vLLM-supported model or any GGUF model.
On T4/P100 models run in fp16, because those GPUs have no bf16.

### Engines

- **vLLM**: highest throughput under many simultaneous users. Needs a T4 or newer.
- **llama.cpp**: built from source in the notebook (~6-10 min the first time) and serves a GGUF file.
  Very reliable on older GPUs; serves `LLAMACPP_PARALLEL` requests at once (default 4 × 32k context).
- **auto**: tries vLLM and falls back to llama.cpp if vLLM fails to start. The `qwen3-coder-30b` preset uses this,
  because vLLM's 4-bit MoE kernels mostly target newer GPUs than the T4.

## Troubleshooting

- **Out of memory**: lower `MAX_NUM_SEQS`, `max_model_len` or `GPU_MEM_UTIL`, or add `--enforce-eager`.
- **vLLM won't start on T4**: pin an older release with `VLLM_VERSION`, or set `VLLM_ATTENTION_BACKEND` to `TRITON_ATTN` / `FLEX_ATTENTION`.
- **llama.cpp out of memory**: lower `LLAMACPP_PARALLEL` or `LLAMACPP_CTX_PER_SLOT`, or use `gguf_quant="Q3_K_M"`.
- **Tool calls arrive as text**: match `tool_parser` to the model (`hermes` for Qwen2.5/Qwen3, `qwen3_coder`, `llama3_json`, `mistral`).
- **Colab disconnected**: re-run all cells. A quick tunnel gets a new URL; ngrok with `NGROK_DOMAIN` keeps the same one.
- Logs live in `/kaggle/working/logs/` on Kaggle and `/content/logs/` on Colab.

## Limits

- **Kaggle**: about 30 GPU-hours per week and 12 hours per session.
- **Colab**: usage limits aren't published and change over time; free sessions end when idle or after at most 12 hours.

Keep your API key private: anyone with the URL and key can use your GPU. Check each platform's terms for your use case.
