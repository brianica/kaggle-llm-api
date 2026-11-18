# Kaggle GPU LLM API

Turn a free Kaggle GPU session (T4 ×2) into a public, password-protected,
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

1. Upload `kaggle_gpu_llm_api.ipynb` to Kaggle (*Create → New Notebook → File → Import Notebook*).
2. *Settings*: Accelerator **GPU T4 ×2**, Internet **On**. (A P100 only works with the llama.cpp backend; vLLM needs a T4 or newer.)
3. *Add-ons → Secrets* (all optional):

   | Secret | Purpose |
   |---|---|
   | `API_KEY` | Key clients must send. Generated and printed if missing. |
   | `NGROK_TOKEN` | ngrok tunnel (recommended; streams properly). `NGROK_DOMAIN` optionally sets your static domain. |
   | `CF_TUNNEL_TOKEN` | Named Cloudflare tunnel pointing to `http://localhost:8000`. Set `CF_HOSTNAME` too. |
   | `HF_TOKEN` | Only for gated models. |

   With no tunnel token, it falls back to a free Cloudflare quick tunnel, which is meant for testing and may not stream.
4. **Run All**. The notebook prints the base URL, API key, model name and copy-paste config for each coding tool.
   To serve without a browser tab open, use **Save Version → Save & Run All (Commit)**, which runs for up to 12 hours.

## Model presets

| Preset | Weights | Engine on T4×2 | Notes |
|---|---|---|---|
| `qwen3-coder-30b` | ~17-19 GB | vLLM, falling back to llama.cpp | **Best for agents**: MoE with 3B active parameters, trained for tool use |
| `qwen2.5-coder-14b` | ~10 GB | split across GPUs | Default |
| `qwen2.5-coder-7b` | ~5.5 GB | one replica per GPU | Most simultaneous users |
| `qwen2.5-coder-32b` | ~19 GB | split across GPUs | Smartest; 16k context |
| `qwen3-8b` | ~6 GB | one replica per GPU | General + coding |

Or set `PRESET = "custom"` with any vLLM-supported model (fp16-compatible; T4 has no bf16) or any GGUF model.

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
- Logs live in `/kaggle/working/logs/`.

## Limits

Kaggle allows about 30 GPU-hours per week and 12 hours per session. Keep your API key private:
anyone with the URL and key can use your GPU. Check Kaggle's terms for your use case.
