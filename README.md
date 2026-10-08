# EduValueBench

Official code and data release for **EduValueBench: Benchmarking Situated Value
Application in Chinese Educational Decision-Making**, accepted as an archival
paper for oral presentation at PlurVA-LLM 2026 @ AACL-IJCNLP 2026.

EduValueBench evaluates how consistently unguided LLM decisions match
expert-authored references grounded in the 12 Core Socialist Values across
Chinese educational contexts. The benchmark contains 1,696 decision-oriented
items spanning seven educational scenario families and three challenge
mechanisms. This repository contains responses from 15 LLMs and the data and
code needed to reproduce the paper's model-level, value-level,
education-family, and evaluator-side results without calling any model API.

- [Paper on OpenReview](https://openreview.net/forum?id=EUzbvS7oZC)
- [Repository](https://github.com/tjunlp-lab/EduValueBench)

## Contents

- `data/eduvaluebench_results.json.gz`: 1,696 prompts, expert-authored
  references, 15 target-model responses, common-panel decisions, and historical
  Hunyuan decisions for the original ten targets.
- `src/analyze_results.py`: offline status-aware aggregation, bootstrap
  intervals, value and education-family breakdowns, sensitivity checks, judge
  audits, and figure generation.
- `src/evaluate.py`: optional resumable evaluation runner for OpenAI-compatible
  endpoints. Credentials are read only from environment variables.
- `expected/`: numerical outputs and figures from the released snapshot.
- `scripts/reproduce.sh`: one-command offline reproduction and verification.

The public release omits evaluator rationales, execution logs, spreadsheets,
API credentials, and worksheet/annotator names. See `data/README.md` for the
field-level schema.

## Offline Reproduction

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/reproduce.sh
```

The command writes regenerated tables and figures to `outputs/`, then compares
the complete JSON summary against `expected/latest_results_summary.json` with a
strict numerical tolerance. It does not use network access or API credentials.

The released snapshot should reproduce these selected headline results:

| Target | Expert-reference consistency |
|---|---:|
| Doubao Seed 1.6 | 54.1% |
| Kimi K3 | 49.1% |
| MiniMax-M3 | 48.3% |
| Qwen3.7-Plus | 45.0% |
| DeepSeek-V4-Flash | 36.3% |
| GLM-5.2 | 31.2% |

The full 15-model table, confidence intervals, both-positive results,
invalid-decision bounds, and deduplicated estimates are regenerated in
`outputs/latest_model_summary.csv`. Derived per-item scores are written to
`outputs/latest_item_scores.csv`, and education-family scores are written to
`outputs/latest_education_family_summary.csv`.

## Evaluation Protocol

The original evaluation has two stages. A target model first answers the
Chinese benchmark question; it is not shown the value label, challenge tag, or
expert reference. A judge then compares that answer with the declared
expert-authored reference and returns a rationale followed by a binary score.
DeepSeek-R1 and Doubao Seed 1.6 form the common panel for all 15 targets.
Historical Hunyuan T1 judgments are retained only for the original ten targets
because that endpoint was unavailable for the five-model extension.

The optional runner intentionally contains no provider keys or account-specific
configuration. Validate a run locally without sending requests:

```bash
python src/evaluate.py \
  --target-model TARGET_MODEL_ID \
  --judge-model JUDGE_MODEL_ID \
  --output runs/target__judge.jsonl \
  --limit 5 \
  --dry-run
```

For an actual run, install `requirements-eval.txt` and set credentials in the
shell environment, never in tracked files:

```bash
export TARGET_API_KEY='...'
export TARGET_BASE_URL='https://provider.example/v1'
export JUDGE_API_KEY='...'
export JUDGE_BASE_URL='https://provider.example/v1'

python src/evaluate.py \
  --target-model TARGET_MODEL_ID \
  --judge-model JUDGE_MODEL_ID \
  --output runs/target__judge.jsonl
```

Use `--answer-field '回答_kimi-k3'` to rejudge a released answer column without
calling a target endpoint. Checkpoints are append-only. A substantive response
is complete when it has a parsed numeric `0` or `1` score; target nonresponses
are recorded as terminal statuses. Items without a reference retain generation
output but are not judged.
The runner writes one JSONL checkpoint per target--judge pair. These checkpoints
are raw run logs; `src/analyze_results.py` consumes the released wide-format
JSON snapshot rather than checkpoint JSONL files directly.

## Interpretation

Scores measure consistency with one declared expert-authored reference, not
universal moral correctness. Target nonresponse is separated from judge
non-completion. Model names identify logged snapshots and should not be treated
as claims about current products.

## Citation

If you use EduValueBench, please cite:

```bibtex
@misc{wu2026eduvaluebench,
  title     = {EduValueBench: Benchmarking Situated Value Application in Chinese Educational Decision-Making},
  author    = {Junru Wu and Yuqi Ren and Yongqi Leng and Zhuowen Han and Xiaoyang Gong and Zheyu Liu and {Jinruwei} and {CongXiao} and Wenjie Tao and {LIU Xiao} and Lei Guan and Deyi Xiong},
  year      = {2026},
  howpublished = {OpenReview},
  url       = {https://openreview.net/forum?id=EUzbvS7oZC},
  note      = {Accepted for oral presentation at PlurVA-LLM 2026 @ AACL-IJCNLP 2026}
}
```

## License

The source code is released under the MIT License. The dataset contains
expert-authored benchmark content and logged third-party model outputs; users
are responsible for complying with applicable provider terms and other rights
when reusing those materials or re-executing model APIs.
