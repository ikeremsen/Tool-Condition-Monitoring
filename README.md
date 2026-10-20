# Tool wear monitoring in milling

Estimating flank wear (VB) of milling inserts from acoustic emission, vibration and spindle motor current, with honest validation.

- **Part A:** NASA / UC Berkeley Milling dataset (`MillNASA/mill.mat`)
- **Part B (planned):** Bosch CNC Machining dataset
- **Sound-only study:** end milling of mild steel, tool condition from a microphone ([Kaggle cutting tool wear audio dataset](https://www.kaggle.com/datasets/nachiketsoni/cutting-tool-wear-audio-dataset))

## Results so far

| Step | Notebook | Read-only version |
|---|---|---|
| 1. Inspect the data | `notebooks/01_inspect_nasa_mill.ipynb` | `reports/01_inspect_nasa_mill.html` |

**Sound-only study:** full write-up in [`reports/02_sound_tool_wear.md`](reports/02_sound_tool_wear.md). Main result: a broken-tooth indicator from the cutting rhythm separates fresh from broken tools at an unseen spindle speed (balanced accuracy 0.63–0.67, chance 0.50). A general 3-class wear model does not transfer between speeds (0.47–0.49, chance 0.33).

The HTML files open in any browser and need no Python. Figures are also saved in `figures/`.

## Project layout

```
MillNASA/        raw data + dataset README (not modified, not in the repository: download "Milling" from the NASA Prognostics Data Repository)
SegmentedData/   sound dataset (not modified, not in the repository: download from Kaggle)
toolwear/        reusable code: one module per dataset, shared plotting style
scripts/         sound study: inspect_dataset.py, spectral_inspection.py, cross_speed_model.py
analysis/        sound study outputs (tables and figures)
notebooks/       one notebook per step, with explanations
reports/         notebooks exported to HTML
figures/         saved figures
```

## Running it yourself

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\jupyter.exe notebook
```

Then open `notebooks/01_inspect_nasa_mill.ipynb` and choose *Run → Run All Cells*.
