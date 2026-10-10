# Verbatim

Lossless decode acceleration for AI coding agents.

Verbatim makes a language model write faster without changing a single output token. It drafts the tokens the model is about to write, from the agent's own session and from a learned drafter, sizes every draft against a measured cost model of the hardware, and has the target model verify the whole draft in one forward pass. In strict mode its kernels make speculative output bit-identical to plain decoding, checked by direct comparison on every prompt.

## Results

Qwen3-8B 4-bit, MacBook Air M3 (16 GB, fanless), MLX. Speedup over stock MLX decoding, pooled over 20 prompts:

| Workload | Verbatim (strict, bit-identical) | Stock DFlash on MLX |
|---|---|---|
| HumanEval (code) | **2.54x** | 1.67x |
| GSM8K (math) | **1.73x** | 1.50x |
| MT-Bench (chat) | 1.02x | 0.94x |
| All | **1.47x** | 1.28x |

Strict outputs: **20 of 20 bit-identical** to plain decoding with the same kernels.

Agent sessions, replayed against the real reference outputs of 10 OpenHands trajectories (SWE-rebench). Tokens accepted per verify pass, which drives speedup on hardware where verification is nearly free:

| Drafter | All turns | `str_replace` edits |
|---|---|---|
| **Verbatim copy drafter** | **2.34** | **4.63** |
| SuffixDecoding (single-path port) | 2.17 | 3.79 |
| AgSpec emission forms | 2.31 | 4.18 |
| Prompt lookup decoding | 2.08 | 2.93 |
| Verbatim copy drafter + held-out corpus | 2.66 | 4.82 |

On llama.cpp with Gemma 4 12B, 30 real SWE-bench agent turns: 1.34x overall and 2.20x on edit turns, 30 of 30 outputs identical, where prompt lookup decoding ran at 0.86x.

## How it works

```
session ──► copy drafter ─────────┐
                                  ├──► controller ──► target verifies [last + draft] ──► keep agreeing prefix
target hidden ──► learned drafter ┘    (measured cost model)
```

1. **Exact kernels.** Batch-invariant Metal kernels for the 4-bit matmul and grouped-query attention compute every row the same way whether the model processes 1 token or 16, so a speculative verify pass and a plain decode step agree bit for bit. Apple's GPU matrix multiply accumulates as sequential fp32 multiply-adds in k order (1.28M of 1.28M probed elements), so a scalar single-row kernel walking K in order reproduces the batched kernel exactly while running at plain-decode speed. Strict mode costs about 21% on single-token steps, below published deterministic modes (34% for SGLang, 61% for batch-invariant kernels from Thinking Machines).
2. **Fast verify.** A tile-major 4-bit weight layout, fused input handling and a 64-key attention block make an 8-token verify at 6k context cost 103 ms against 193 ms for stock MLX, while one copy of the weights lives in memory (6 GB active for an 8B model).
3. **Copy drafter.** Agents spend much of their output re-typing text they have read. The drafter indexes the session as bytes, plus each line's line-number-stripped, diff-prefixed, JSON-escaped and re-indented forms, finds the longest suffix match, and retokenizes the continuation across the token boundary so the draft is exactly what the model would emit.
4. **Learned drafter.** For text with nothing to copy, DFlash's block-diffusion drafter proposes a block of tokens from the target's hidden states.
5. **Controller.** Each step takes the copy draft when one exists, otherwise a learned draft when its measured throughput beats plain decoding, otherwise a plain step. Verify width comes from a cost table measured at short and long context.

## Equivalence

Greedy speculative decoding must emit exactly the tokens the target would have emitted alone. The benchmarks compare every generated token ID against a plain run of the same prompt. Strict mode extends this to bit-identity under the same kernels: the batched verify pass and the single-token step share one reduction order.

## Reproduce

Hardware used: MacBook Air M3, 16 GB, macOS 26.6.

```bash
uv venv --python 3.12 .venv-dflash
uv pip install --python .venv-dflash/bin/python mlx mlx-lm dflash pyarrow
.venv-dflash/bin/hf download mlx-community/Qwen3-8B-4bit --local-dir models/mlx-qwen3-8b-4bit
.venv-dflash/bin/hf download z-lab/Qwen3-8B-DFlash-b16 --local-dir models/dflash-qwen3-8b

.venv-dflash/bin/python verbatim/qmv.py
.venv-dflash/bin/python verbatim/attn.py
.venv-dflash/bin/python scripts/dflash_bench.py --stock --configs base --out results/stock.jsonl
.venv-dflash/bin/python scripts/dflash_bench.py --configs sbase,sadapt,sadaptc --out results/verbatim.jsonl
.venv-dflash/bin/python scripts/simulate.py ds/nebius/rg0_400.parquet --format openhands --trajs 10 --procs 1 \
    --configs verbatim,suffix,agspec-forms,pld --cost data/cost_h100_roofline.json --out results/sim.jsonl
```

The llama.cpp path (`scripts/bench.py`, `verbatim/llama_engine.py`) reproduces the Gemma 4 12B agent-turn results.

## Data

- [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories), MIT. Claude 3.7 Sonnet sessions with text-embedded `str_replace_editor` calls.
- [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories), CC-BY-4.0. OpenHands sessions with JSON tool calls.

## Next milestone

About 36% of agent output tokens appear nowhere in the context, which caps every copy-based drafter near 2.8 tokens per pass on these traces. The next step is a learned drafter fine-tuned on long agent trajectories to cover that share, then a head-to-head on an H100 against SuffixDecoding, AgSpec, EAGLE-3 and DFlash on the same agent sessions.
