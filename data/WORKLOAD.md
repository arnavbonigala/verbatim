# Workload
- Source: SWE-bench/SWE-smith-trajectories (Hugging Face), split `tool`, shard tool-00004-of-00008.parquet.
- License: MIT (dataset card `license: mit`).
- Selection: claude-3-7-sonnet-20250219 trajectories, one target assistant turn per trajectory, seeded shuffle (seed 0), prompt capped at 7000 Qwen2.5 tokens by dropping oldest middle messages.
- Counts: str_replace=12, create=6, after_obs=6, reasoning=6 (total 30).
- Prompt tokens min/median/max: 1620/6150/6974.
