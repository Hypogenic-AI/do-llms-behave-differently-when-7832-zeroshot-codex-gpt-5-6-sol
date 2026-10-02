# Who Wrote the Prompt?

This repository tests whether Qwen2.5-7B-Instruct behaves differently when the same request is rendered in ordinary human-chat style versus polished, AI-assistant-like style.

## Main result

The style manipulation was real and strongly visible inside the model: a held-out mean-difference probe reached AUROC 0.978. It was nevertheless almost orthogonal to a separately constructed evaluation-awareness direction (maximum absolute cosine 0.072), and it produced no clear behavioral shift:

- MMLU correct-answer probability: −2.1 percentage points for LLM style versus human style (95% paired bootstrap CI −10.4 to +5.7; 39 items).
- Sycophantic-answer probability: −0.2 points (CI −9.9 to +9.9; 40 items).
- Harmful-request refusal: 87.5% under both styles (16 complete judged pairs).
- Benign over-refusal: 18.5% versus 14.8% (difference −3.7 points, CI −11.1 to 0; 27 pairs).

With the prompt text held fixed, adding the learned style direction at transformer block 20 left accuracy and all refusal labels essentially unchanged. A small sycophancy movement was not selective relative to a norm-matched orthogonal random direction. The defensible conclusion is therefore a bounded null for this model: “reads LLM-written” is decodable, but we found little evidence that the measured linear feature materially controls these behaviors.

The finished paper is [paper_draft/main.pdf](paper_draft/main.pdf), with source in [paper_draft/main.tex](paper_draft/main.tex).

## Study design

We began with 40 MMLU questions, 40 Anthropic model-written political sycophancy items, and 30 harmful plus 30 matched benign JailbreakBench requests. Each source received two rewrites in one API call, holding the rewriting model fixed within item. A blinded semantic-equivalence screen retained 124 evaluation items. A disjoint 126-item screened set was used for activation analysis.

The experiments comprise:

1. A direct H/A style-classification manipulation check.
2. Paired textual comparisons for accuracy, sycophancy, harmful refusal, and benign over-refusal.
3. Held-out activation projection at seven layers.
4. Cosine comparison to an independent evaluation-awareness direction built from 204 public contrastive prompts.
5. Fixed-text residual-stream interventions in both style directions, plus both signs of an orthogonal norm-matched random vector.

All inference is deterministic. Statistical intervals use 10,000 paired bootstrap resamples with seed 7832.

## Repository layout

- `src/prepare_prompts.py`: deterministic benchmark sampling and paired rewriting.
- `src/validate_rewrites.py`: blinded semantic-equivalence and style check.
- `src/run_local_model.py`: local behavior, activation extraction, probing, and steering.
- `src/judge_safety.py`: refusal/compliance judgments.
- `src/analyze_results.py`: statistics, tables, figures, and machine-readable summary.
- `results/`: prompts, raw generations/judgments, activations, vectors, tables, and `summary.json`.
- `paper_draft/`: LaTeX, bibliography, figures, and compiled PDF.
- `data/`, `models/`, and `cache/`: ignored download/cache directories.

Files whose names contain `interleaved`, `mixed_steering`, or `with_duplicates` are audit logs preserved from a detected concurrent-writer issue. The canonical analyzed files are `results/local_model_outputs.jsonl` and `results/safety_judgments.jsonl`; both contain unique, screened records from the clean direction construction.

## Reproduction

Hardware used here: one NVIDIA RTX A6000 (48 GB), 32 CPU cores, Python 3.12.8, CUDA-enabled PyTorch 2.14.1. API calls require `OPENROUTER_KEY`; model and dataset downloads may use `HF_TOKEN` if configured. No credential is stored by the code.

Create the isolated environment:

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.uvpython" UV_PYTHON_PREFERENCE=managed uv python install 3.12
UV_PYTHON_INSTALL_DIR="$PWD/.uvpython" UV_PYTHON_PREFERENCE=managed uv venv --python 3.12 .venv
UV_PYTHON_INSTALL_DIR="$PWD/.uvpython" uv pip sync --python .venv/bin/python requirements-lock.txt
```

Download Qwen2.5-7B-Instruct and the small public evaluation-awareness repository:

```bash
HF_HOME="$PWD/cache/huggingface" HF_HUB_CACHE="$PWD/models" .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='models')"
mkdir -p data/eval_awareness_repo
curl -L https://github.com/Jordine/evaluation-awareness-probing/archive/refs/heads/main.tar.gz -o data/eval_awareness_repo.tar.gz
tar -xzf data/eval_awareness_repo.tar.gz -C data/eval_awareness_repo --strip-components=1
```

Run the pipeline:

```bash
HF_HOME="$PWD/cache/huggingface" .venv/bin/python src/prepare_prompts.py --workers 8
.venv/bin/python src/validate_rewrites.py --workers 8
CC="$PWD/src/zig_cc" CXX="$PWD/src/zig_cxx" HF_HOME="$PWD/cache/huggingface" CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/run_local_model.py
.venv/bin/python src/judge_safety.py --workers 8
MPLBACKEND=Agg .venv/bin/python src/analyze_results.py
(cd paper_draft && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
```

On systems where Triton needs its runtime extension compiled but `libcuda.so.1` is not found through the compiler search path, link the system library into `.venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/` before local inference. The provided Zig compiler wrappers avoid requiring a system compiler. A normal GCC/Clang installation can instead be passed through `CC`/`CXX`.

API rewriting and judging are resumable: existing item IDs or experiment keys are skipped. To reproduce the exact historical mixture of rewriters, use the checked-in `results/prompts.jsonl`; a fresh full rewrite run with one endpoint may differ stylistically.

## Notes on interpretation

“Human” and “LLM” in this project mean induced writing styles, not verified authorship: both treatments were generated by an LLM to isolate style from generator identity within each pair. The LLM-style texts were longer on average (67.1 versus 40.4 words), so textual comparisons cannot fully separate authorship cues from surface form. The activation intervention holds text exactly fixed, but the learned direction may itself encode length, formality, and structure. Results concern one 7B model and should not be generalized to frontier systems without replication.
