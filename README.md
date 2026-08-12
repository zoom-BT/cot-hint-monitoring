# cot-hint-monitoring

Study of **chain-of-thought (CoT) faithfulness** on multiple-choice questions: when a model gets an **answer hint** (correct or wrong), or is told its **reasoning will be monitored**, does it change its final answer without saying so in the CoT?

This sits in **mechanistic interpretability / model biology** for AI safety: CoT monitoring is often proposed as a way to catch suspicious reasoning. The experiment tests when that assumption breaks.

Default model: `Qwen/Qwen2.5-7B-Instruct` (4-bit, inference only).

## Context

Work in the line of:

- **Neel Nanda** (Google DeepMind) — pragmatic interpretability, model biology, CoT as a safety-relevant signal ([MATS](https://www.matsprogram.org/), [mentor page](https://www.matsprogram.org/mentor/neel))
- **Chen et al.** — models can follow hints without admitting them in the CoT
- **Arcuschin et al.** — unfaithful CoT in the wild (rationalization, last-minute answer flips, shortcuts)

Related questions: eval awareness, sycophancy, whether “read the CoT” is a reliable monitor.

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
