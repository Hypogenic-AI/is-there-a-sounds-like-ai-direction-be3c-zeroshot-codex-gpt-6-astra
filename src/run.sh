#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/models/hf"
export TOKENIZERS_PARALLELISM=false
mkdir -p results data models paper_draft/tables paper_draft/figures
P=.venv/bin/python
$P src/setup_data.py
$P src/prepare.py
$P src/make_calibration.py
$P src/experiment.py extract --kind instruct
$P src/experiment.py extract --kind base
$P src/readout.py
$P src/base_comparison.py
$P src/experiment.py controls
$P src/experiment.py steer
$P src/transfer_steer.py
$P src/quality_audit.py
$P src/local_judge.py
$P src/detect.py
$P src/modern_detector.py
$P src/quality_audit.py
$P src/analyze.py
$P src/supplement.py
$P src/analyze_transfer.py
$P src/analyze_modern.py
if [[ "${RUN_API_AUDIT:-0}" == "1" ]]; then
  $P src/judge_api_confirm.py
fi
if [[ ! -f results/claude_judge_raw.jsonl ]]; then
  echo "Final manuscript includes a late API audit: restore saved results/claude_judge_raw.jsonl or set RUN_API_AUDIT=1 with OPENROUTER_KEY."
  exit 1
fi
$P src/analyze_claude.py
$P src/validate.py
$P src/build_tables.py
(cd paper_draft && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex)

$P src/audit_artifacts.py
