# Execution notes
- Hardware inspection: one idle NVIDIA RTX A6000 (49140 MiB total, 1 MiB used), 32 CPU threads, roughly 188 GiB available host RAM at session start.
- Environment created under `.venv/` with `uv`; nothing installed into system Python.
- Initial latest PyTorch 2.14.1 attempted a Triton compilation and failed because no C compiler was present. Replaced with PyTorch 2.6.0+cu124. Final inference uses Transformers 4.51.3 and bfloat16 CUDA.
- Raw HC3 JSONL has no `id` column; IDs are zero-based original line indices, preserving stable source references.
- Open-domain QA did not meet the requested balanced-pair count at 96 tokens; protocol amended before outcomes to four domains.
- Exact excerpt deduplication added after integrity check found two train/test overlaps; both model activation extractions rerun before readout inspection or steering.
- Original API judge failed: OpenRouter 403 daily limit; fallback OpenAI 401 invalid key. Only sanitized failure types retained, no credentials or API error bodies in artifacts. The initial judge was local Mistral. After UTC-day rollover, a quota recheck succeeded and a late Claude audit scored the same fixed outputs; both sets of ratings are retained.
- Qwen Transformers emits an SDPA sliding-window warning; the checkpoint configuration has `use_sliding_window: false`. Generation emits warnings about stored sampling parameters; `do_sample=False` means those stored temperature/top-p/top-k values are ignored.
- Final scientific conclusions must use only completed raw outputs and explicitly distinguish imperfect automated quality ratings from human factual review.

- Final API audit: all 1,570 leading JSON rating objects are complete and schema-valid; the service often appended explanations despite the JSON-only instruction, and 1,492 responses hit the output-token cap after the complete JSON. Raw responses preserve that behavior. Provider-reported usage and costs are recorded in api_usage.json.
