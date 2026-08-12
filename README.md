# CoT faithfulness under hints and monitoring

Does a language model's **chain of thought stay faithful** when it is given an **answer hint**, or told that its **reasoning is being monitored**?

If the model changes its final answer but the CoT never mentions the hint (or the monitoring), the CoT is a bad safety signal. This repo measures that on multiple-choice questions.

Default model: `Qwen/Qwen2.5-7B-Instruct` (4-bit, inference only).

## Why this question

**Chain-of-thought monitoring** is a proposed way to catch deceptive or sketchy reasoning. That only works if the CoT is *faithful*: it actually reports the reasoning that produced the answer.

This is **model biology / pragmatic interpretability** for AI safety, in the line of:

- **Neel Nanda** (Google DeepMind, mechanistic interpretability) and [MATS](https://www.matsprogram.org/mentor/neel)
- **Chen et al.** — models follow hints without admitting them in the CoT
- **Arcuschin et al.** — unfaithful CoT in the wild (rationalization, last-minute flips, shortcuts)

Related: eval awareness, sycophancy, whether “just read the CoT” is a reliable monitor.

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
git clone https://github.com/zoom-BT/cot-faithfulness-under-hints.git
cd cot-faithfulness-under-hints
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
