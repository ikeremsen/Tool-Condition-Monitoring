# Tool Condition Monitoring: can a computer hear a worn milling cutter?

Experienced machinists can often *hear* when a cutting tool is worn or broken. This project tests whether a computer can do the same from a microphone recording, and checks the results honestly instead of chasing a high accuracy number.

**Main study:** [Tool condition from sound only (full report)](reports/02_sound_tool_wear.md)

---

## The project in one minute

**The problem.** In milling, a rotating cutter with several cutting edges removes metal from a workpiece. The edges wear down and can break. A worn tool gives poor surfaces and inaccurate parts; a broken one can scrap the part. Stopping to inspect the tool costs time, so factories want to detect tool condition *while the machine is running*.

**The data.** 1488 ten-second sound recordings of an end mill (16 mm, 4 cutting edges) cutting mild steel at two spindle speeds (520 and 635 rpm). Each recording is labelled fresh, moderately worn, broken, or machine running without cutting.

**The key idea.** Each cutting edge hits the steel once per revolution, so the cutting noise *pulses in a rhythm*, like a hammer striking 4 times per turn. If one edge is broken, the 4 hits are no longer equal, and the rhythm changes from "4 beats per turn" towards "1 beat per turn". We measure that rhythm in the sound.

![The cutting rhythm found in the sound](analysis/spectral/2_envelope_spectrum_635.png)

*Envelope spectrum at 635 rpm: how often the cutting noise pulses. The peak at 40 Hz is exactly 4 × the measured spindle speed (10 Hz), which confirms the 4 cutting edges. The grey curve is the machine running without cutting.*

**What we found.**

| Question | Answer |
|---|---|
| Does the sound contain the cutting physics? | **Yes**: the 4-edge rhythm is clearly visible, at 4 × the real spindle speed |
| Can a model trained at one speed tell **fresh from broken** at the other speed? | **Partly**: 63–67 % correct with one physical "broken-tooth" indicator (guessing = 50 %) |
| Can it tell **fresh / moderate / broken** apart at a new speed? | **Not well**: 47–49 % correct (guessing = 33 %) |
| Is more data needed to make strong claims? | **Yes**: the dataset has only one physical tool per wear class (see below) |

## Why "honest" validation matters here

Two traps would make the results look much better than they are, and both were avoided:

1. **One tool per class.** All "fresh" recordings come from tool A, all "broken" from tool C. A model can learn to recognise *which tool* it hears instead of *how worn* it is. No test with this dataset can fully rule that out, so the claims above are kept modest.
2. **Recordings cut into pieces.** The 10 s files were cut from longer recordings. Mixing pieces of the same recording into both training and testing makes a model look better than it is. All tests here keep each original recording together, and the main test uses a **speed the model has never seen**.

A [published study](https://www.mdpi.com/2075-1702/14/1/59) using recordings at the same two speeds (apparently this dataset) reports 94–97 % accuracy, with each speed trained and tested separately. The gap shows how much the test design matters.

## How it was done

| Step | What | Output |
|---|---|---|
| 1. Check the data | Is every file readable, correctly labelled, same format? | [`scripts/inspect_dataset.py`](scripts/inspect_dataset.py) → `analysis/` |
| 2. Find the physics | Where in the sound is the cutting? Is the 4-edge rhythm there? | [`scripts/spectral_inspection.py`](scripts/spectral_inspection.py) → `analysis/spectral/` |
| 3. Test across speeds | 6 physical indicators, simple model, trained at 520 rpm, tested at 635 rpm | [`scripts/cross_speed_model.py`](scripts/cross_speed_model.py) → `analysis/cross_speed/` |

Every step, figure and number is explained in the **[full report](reports/02_sound_tool_wear.md)**.

---

## Other part of the project: NASA milling dataset

Estimating flank wear (the worn-away band on the cutting edge, in mm) from vibration, acoustic emission and spindle motor current, using the NASA / UC Berkeley milling dataset. Step 1, data inspection, is done: [notebook](notebooks/01_inspect_nasa_mill.ipynb) (GitHub shows it with its figures). A Bosch CNC machining dataset is planned as a further part.

## Repository layout

```
reports/         write-ups: start with 02_sound_tool_wear.md
scripts/         sound study, one script per step
toolwear/        reusable code (sound indicators, NASA loader, plot style)
analysis/        sound study outputs: tables (.csv) and figures (.png)
notebooks/       NASA study notebook
figures/         NASA study figures
```

The datasets themselves are not included (too large, and not ours to redistribute):

- Sound data: [Cutting Tool Wear Audio Dataset (Kaggle)](https://www.kaggle.com/datasets/nachiketsoni/cutting-tool-wear-audio-dataset). Put it in `SegmentedData/520/...` and `SegmentedData/635/...`.
- NASA data: "Milling" from the NASA Prognostics Data Repository. Put `mill.mat` in `MillNASA/`.

## Running it yourself

Python 3.13 on Windows (adapt the paths on macOS/Linux):

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# sound study (steps 2 and 3 take about 5 minutes each)
.venv\Scripts\python.exe scripts\inspect_dataset.py
.venv\Scripts\python.exe scripts\spectral_inspection.py
.venv\Scripts\python.exe scripts\cross_speed_model.py

# NASA study
.venv\Scripts\jupyter.exe notebook notebooks\01_inspect_nasa_mill.ipynb
```
