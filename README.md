# Is there a “sounds like AI” direction in the residual stream?

A controlled research study of authorship readout and generation interventions in Qwen2.5-1.5B. The paper is built from executed local experiments, with raw generations and evaluator outputs retained.

## Main findings

- A mean residual direction reaches **0.992 AUROC** on held-out HC3 and about **0.89** on two HAP-E generators. Base and instruct representations perform similarly.
- At strengths +1 versus −1, the HC3 direction changes RoBERTa's score by **0.248 [0.149, 0.355]**. An effect remains after removing estimated formality, persona, length, likelihood, and domain directions.
- A supplemental contemporary detector corroborates continuous score shifts, but **58/60 negatively steered answers remain detected**. Extreme negative steering lowers rated answer adequacy.
- The local Mistral style judge fails provenance discrimination. The late Claude audit achieves AUROC 0.856/0.849 on HC3/HAP-E and finds a style effect, but this attenuates after nuisance removal and is unresolved on the small quality-qualified subset. A direction fit on separate HAP-E documents moves the two trained detectors in **opposite directions**.

The evidence supports a steerable detection-related writing component, not a unique, content-preserving, human-perceived “AI-ness” axis. See [the compiled paper](paper_draft/main.pdf) for confidence intervals, controls, and limitations.

## Study design

- **Representation data:** 500 training, 100 validation, and 200 held-out HC3 question pairs from four domains. Human and ChatGPT excerpts are normalized, deduplicated, and exactly 96 Qwen tokens long.
- **Transfer:** 180 HAP-E documents, paired human continuations with GPT-4o-mini and Llama-3-8B-Instruct continuations.
- **Models:** Qwen2.5-1.5B base and instruct. Mean-difference and logistic readouts after blocks 7, 14, and 21. Validation AUROC selects the intervention block.
- **Causal test:** 60 held-out questions × 19 conditions = 1,140 greedy answers. Authorship strength sweep, two equal-norm random controls, formality and persona-proxy controls, nuisance orthogonalization, centered ablation, and a human-style prompt.
- **Evaluation:** frozen RoBERTa GPT-2 detector, a blinded local Mistral-7B-Instruct-v0.3 judge, an exploratory Desklib DeBERTa detector, and a late blinded Claude Haiku 4.5 audit. Quality, length, likelihood, and fixed-length detector checks accompany paired bootstrap intervals. All four evaluators are audited on held-out human/machine excerpts.

The exploratory cross-corpus replication adds 120 generated answers. The complete run contains **1,420 Qwen-generated answers** (1,140 primary + 120 transfer + 160 nuisance-control answers) and **1,570 Mistral evaluations plus 1,570 late Claude evaluations** (1,260 experimental answers + 250 natural-text calibration excerpts + 60 constructed quality-audit controls). No human judgments were collected.

## Artifacts

- `paper_draft/main.tex`, `paper_draft/main.pdf`: paper source and compiled manuscript.
- `src/`: data preparation, extraction, steering, judging, analysis, integrity checks, and table generation.
- `results/protocol.md`: pre-outcome design and documented feasibility/infrastructure amendments.
- `results/split_manifest.jsonl`, `results/ood_manifest.jsonl`: exact data membership.
- `results/features_*.npz`, `results/directions.npz`: measured activation summaries and intervention vectors.
- `results/generations.jsonl`, `results/control_generations.jsonl`: raw generated text.
- `results/judge_raw.jsonl`, `results/detector_scores.jsonl`: raw independent evaluations.
- `results/*.csv`: numerical analyses; tables and figures are generated from these.
- `results/revisions.json`, `requirements.lock.txt`: pinned dependencies and source revisions.

## Reproduce

Use a CUDA GPU with sufficient free memory (the run used one otherwise idle RTX A6000, 48 GB), Python 3.12, `uv`, and a LaTeX installation with `pdflatex` and `bibtex`.

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.lock.txt
bash src/run.sh
```

Run from the repository root. Downloaded models and datasets go under ignored `models/` and `data/`. The models and datasets are public; the local experiments require no API key. A late external Claude audit requires `OPENROUTER_KEY` if recomputed; its saved raw responses support offline reanalysis. OpenRouter exhausted its daily key limit and the provided fallback OpenAI key was invalid; these unsuccessful attempts are recorded separately. After UTC-day rollover, OpenRouter recovered and the unchanged original Claude rubric was used for an additional audit of the fixed outputs. `src/judge.py` preserves the API rubric/client; `src/local_judge.py` runs Mistral, and `src/judge_api_confirm.py` runs the late Claude audit. The pipeline reuses saved Claude responses by default; set `RUN_API_AUDIT=1` to obtain new ones.

Generation and judging resume by completed condition or sample key. For a fully fresh run, first back up and move the existing `results/` directory. The scripts contain the pinned revisions; preserve `results/revisions.json` with your backup for provenance. Running the full pipeline creates a new results directory. Restore the saved `claude_judge_raw.jsonl` there for offline reuse, or run `RUN_API_AUDIT=1 bash src/run.sh` with an authorized `OPENROUTER_KEY` to recompute the late audit. The final paper includes that audit, so rebuilding every result from scratch requires either its archived responses or API access. Reusing raw outputs permits analysis-only regeneration:

```bash
.venv/bin/python src/analyze.py
.venv/bin/python src/analyze_transfer.py
.venv/bin/python src/analyze_modern.py
.venv/bin/python src/analyze_claude.py
.venv/bin/python src/build_tables.py
cd paper_draft
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Analysis-only regeneration needs `data/calibration.jsonl`, `data/corpus.jsonl`, and `data/ood.jsonl`; recover them with `setup_data.py`, `prepare.py`, and `make_calibration.py` if the ignored data directory was not preserved. The full run reconstructs them automatically.

## Interpretation and provenance

This is an exploratory single-family, single-size study with automatic evaluation, not human authorship verification. A high probe AUROC is not evidence that the corresponding direction is a unique style control. Quality-filtered comparisons condition on post-treatment outcomes and are descriptive. Our assistant/storyteller direction is a proxy, not the published Assistant Axis. Base-model generation and larger checkpoints were not tested.

HC3 is credited to Guo et al. and distributed under CC-BY-SA-4.0; the HAP-E dataset card specifies MIT and credits Reinhart et al. Source texts are downloaded rather than bundled. Raw generated results include HC3 questions; their source attribution and license continue to apply. Model license details reside in their pinned public model cards. Literature sources and scope distinctions are documented in `results/literature_notes.md` and the paper bibliography.

The implementation, analyses, and manuscript were produced autonomously by an AI agent. All experimental numbers come from the recorded runs. No human or expert factual ratings were collected.
