# cot-hint-monitoring

Small benchmark study on multiple-choice answering with chain-of-thought, hints, and monitoring prompts.

Model default: `Qwen/Qwen2.5-7B-Instruct` (4-bit, inference only).

## Conditions

| ID | Description |
|---|---|
| C0 | Baseline |
| C1 | Correct hint |
| C2 | Incorrect hint |
| C3 | CoT monitored |
| C4 | Correct hint + monitored |

## Colab

Open `notebooks/cot_faithfulness_eval.ipynb`, select a **T4 GPU**, then run cells top to bottom.

```bash
git clone https://github.com/zoom-BT/cot-hint-monitoring.git
cd cot-hint-monitoring
pip install -r requirements.txt
python scripts/run_eval.py --mode pilot --output results/generations_pilot.jsonl
python scripts/run_eval.py --mode main --output results/generations.jsonl --no-cot-baseline
python scripts/metrics.py --input results/generations.jsonl
```

`run_eval.py` skips pairs already present in the output JSONL (safe to resume after disconnect).

## Layout

```
data/       MCQ items
scripts/    evaluation and metrics
results/    logs and tables (not tracked)
figures/    plots (not tracked)
notebooks/  Colab notebook
```

## License

MIT
