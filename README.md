# Verbatim

Lossless speculative decoding for AI agents.

Coding agents spend much of their output re-typing text they have already seen: the `old_str` of an edit copied from a file view, the unchanged lines of a replacement block, a file rewritten with one function changed, a command re-run after a failure, file contents JSON-escaped into a tool call. Verbatim indexes the agent's session and drafts those spans for the target model, which verifies the whole draft in one forward pass. Greedy output is token-for-token identical to normal decoding. The model just writes faster.

<!-- RESULTS -->

## How it works

```
session tokens ──► text history ──► emission forms ──► longest suffix match ──► continuation text
                                                                                      │
draft token IDs ◄── boundary-safe retokenization ◄── cost-aware draft length ◄────────┘
       │
       ▼
target model verifies [last token + draft] in one batch, keeps the longest agreeing prefix
```

1. **Session index.** Every token the session has seen (system prompt, issue, tool output, the model's own earlier turns) is kept as bytes. A draft request finds the longest suffix of the current output that occurs earlier in the session, using a binary search over match length with `bytes.rfind`.
2. **Emission forms.** Agents rarely copy text in the form they read it. A `cat -n` view carries line numbers, a diff prefixes every line with `+`, `-` or a space, a JSON tool call escapes every newline and quote, and a moved block is re-indented. As complete lines arrive, Verbatim also indexes each line-numbers-stripped, diff-prefixed, JSON-escaped and re-indented rendering, so a copy can be found in whichever form the model is writing.
3. **Text to tokens.** Matching happens on text, then the continuation is retokenized together with the last two generated tokens. If the joint tokenization reproduces those tokens, the remainder is exactly what the model would emit at that position; the final token is dropped because it may be cut mid-word. This handles the token-boundary mismatch that breaks naive token-level copying.
4. **Cost-aware adaptive length.** Verify cost on a GPU is not linear in draft length. On Apple Metal a 2 to 8 token batch costs nearly one forward pass per token, while 10 to 32 tokens cost a flat ~4 to 6 passes. Verbatim keeps a per-batch-size cost table (profiled once, then updated online from measured timings) and online acceptance statistics bucketed by match length, and picks the draft length that maximizes expected tokens per second. A 300-byte match in a file copy drafts 31 tokens; a 12-byte coincidence in prose drafts nothing.
5. **Engine-agnostic core.** `verbatim/drafter.py` depends only on a `tokenize` and `detokenize` function. Its `__call__(input_ids)` matches llama-cpp-python's `LlamaDraftModel` signature, and `propose(input_ids)` returns one or more candidate drafts. `verbatim/llama_engine.py` is the only engine-specific code: a greedy verify loop on llama.cpp's low-level batch API.

## Equivalence

Greedy speculative decoding must emit exactly the tokens the target would have emitted alone. The benchmark checks this on every turn by comparing every generated token ID against a no-speculation run of the same prompt. The harness earned its keep during development: it caught an engine bug where a copied end-of-turn token was accepted and generation continued past it.

## Reproduce

Hardware used: MacBook Air M3, 16 GB, macOS 26.6, llama-cpp-python 0.3.36 with Metal.

```bash
uv venv --python 3.12 .venv
CMAKE_ARGS="-DGGML_METAL=on -DLLAMA_OPENSSL=OFF" uv pip install --python .venv/bin/python llama-cpp-python huggingface_hub pyarrow matplotlib
.venv/bin/hf download google/gemma-4-12B-it-qat-q4_0-gguf gemma-4-12b-it-qat-q4_0.gguf --local-dir models
.venv/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir models
.venv/bin/hf download SWE-bench/SWE-smith-trajectories data/tool-00004-of-00008.parquet --repo-type dataset --local-dir ds

.venv/bin/python tests/test_drafter.py
.venv/bin/python scripts/build_workload.py ds/data/tool-00004-of-00008.parquet
.venv/bin/python scripts/bench.py --model models/gemma-4-12b-it-qat-q4_0.gguf --configs none,pld,verbatim,verbatim-noforms --out results/gemma4_12b
.venv/bin/python scripts/report.py results/gemma4_12b
.venv/bin/python scripts/simulate.py ds/data/tool-00004-of-00008.parquet --trajs 120 --out results/sim_q4km.jsonl
.venv/bin/python scripts/sim_report.py results/sim_q4km.jsonl
.venv/bin/python scripts/build_demo.py models/gemma-4-12b-it-qat-q4_0.gguf results/gemma4_12b results/qwen7b_q4km
cd demo && python -m http.server
```

On an Intel Homebrew machine, an x86 OpenSSL can leak into the arm64 build; `-DLLAMA_OPENSSL=OFF` avoids it.

## Data

- [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), MIT. Claude 3.7 Sonnet sessions with text-embedded `str_replace_editor` calls.
- [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), CC-BY-4.0. OpenHands sessions with JSON tool calls.

## Next milestone

This is a proof of concept on llama.cpp on a laptop. The drafter is engine-agnostic: the vLLM and SGLang integration replaces only `llama_engine.py` with a proposer that hands `propose()` drafts to the engine's speculative verifier. Batch-invariant verification kernels on NVIDIA GPUs, which make batched and single-token logits bit-identical, are the other half of the production story. Both need GPU resources.
