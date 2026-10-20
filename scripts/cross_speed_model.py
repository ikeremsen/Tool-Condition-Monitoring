"""Step 3: do the physical indicators work at a speed the model has never seen?

1. Compute 6 indicators for every full-length file (toolwear/sound_features.py).
2. Plot each indicator per class at both speeds (does the class order stay the same?).
3. Train a simple model at 520 rpm, test it at 635 rpm (and the reverse, as a check).
4. Compare with a same-speed test where whole recordings ("takes") are held out.

Classes: fresh / moderate / broken. BASE (machine only) is left out of the model: it is
not a wear state and is trivially easy to recognise. Chance level = 33 %.

Outputs in analysis/cross_speed/:
    features.csv                one row per file (cached; delete it to recompute)
    1_indicators_by_class.png   indicator distributions, 520 vs 635 side by side
    2_confusion_matrices.png    where the cross-speed model is right / wrong
    results.txt                 all numbers printed below

Run from the project folder:
    .venv\\Scripts\\python.exe scripts\\cross_speed_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from toolwear.plotting import use_project_style, SERIES, TEXT_MUTED
from toolwear.sound_features import (FEATURES, load_left, raw_features, add_base_reference, assign_takes)
use_project_style()

INVENTORY = ROOT / "analysis" / "inventory.csv"
OUT_DIR = ROOT / "analysis" / "cross_speed"
WEAR_CLASSES = ["fresh", "moderate", "broken"]
COLORS = {"base": TEXT_MUTED, "fresh": SERIES[0], "moderate": SERIES[1], "broken": SERIES[2]}
FEATURE_LABELS = {
    "hf_energy_db": "10-20 kHz energy vs BASE (dB)",
    "spectral_centroid": "spectral centroid (kHz)",
    "hf_kurtosis": "10-20 kHz kurtosis (-)",
    "order1_db": "once-per-revolution pulse (dB)",
    "order4_db": "tooth-passing pulse (dB)",
    "order4_minus_1_db": "tooth minus revolution pulse (dB)",
}

results = []


def report(text=""):
    print(text)
    results.append(str(text))


def compute_features():
    inv = assign_takes(pd.read_csv(INVENTORY))
    full = inv[inv["frames"] == inv["frames"].mode()[0]].reset_index(drop=True)
    rows = []
    for i, r in full.iterrows():
        x, sr = load_left(ROOT / r["path"])
        rows.append({"file": r["file"], "rpm": r["rpm"], "wear_class": r["wear_class"], "tool": r["tool"],
                     "sample": r["sample"], "take_id": r["take_id"], **raw_features(x, sr, r["rpm"])})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(full)} files")
    return add_base_reference(pd.DataFrame(rows))


def plot_indicators(df):
    """One panel per indicator: boxes for each class, 520 rpm (solid) next to 635 rpm (light)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    classes = ["base"] + WEAR_CLASSES
    for ax, feat in zip(axes.flat, FEATURES):
        pos, data, cols, alphas, labels = [], [], [], [], []
        for i, cls in enumerate(classes):
            for j, rpm in enumerate([520, 635]):
                pos.append(i * 3 + j)
                data.append(df.loc[(df["wear_class"] == cls) & (df["rpm"] == rpm), feat])
                cols.append(COLORS[cls]); alphas.append(0.75 if rpm == 520 else 0.3)
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, c, a in zip(bp["boxes"], cols, alphas):
            patch.set_facecolor(c); patch.set_alpha(a)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(classes))], classes)
        ax.set_title(FEATURE_LABELS[feat])
    fig.suptitle("Indicators per class: left box = 520 rpm (dark), right box = 635 rpm (light). "
                 "A useful indicator keeps the same class order at both speeds.", fontsize=10)
    fig.savefig(OUT_DIR / "1_indicators_by_class.png")
    plt.close(fig)


def make_model():
    # Standardise each indicator, then a linear model: few parameters, easy to read, hard to overfit.
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def cross_speed(df, train_rpm, test_rpm, features):
    tr, te = df[df["rpm"] == train_rpm], df[df["rpm"] == test_rpm]
    model = make_model().fit(tr[features], tr["wear_class"])
    pred = model.predict(te[features])
    return balanced_accuracy_score(te["wear_class"], pred), pred, model


def same_speed_by_take(df, rpm, features, n_splits=5):
    """Within one speed, hold out whole takes, so neighbouring 10 s pieces never sit on both sides."""
    d = df[df["rpm"] == rpm]
    pred = cross_val_predict(make_model(), d[features], d["wear_class"],
                             groups=d["take_id"], cv=GroupKFold(n_splits=n_splits))
    return balanced_accuracy_score(d["wear_class"], pred)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_file = OUT_DIR / "features.csv"
    if feat_file.exists():
        df = pd.read_csv(feat_file)
        print(f"Loaded cached {feat_file.name}")
    else:
        print("Computing indicators for every full-length file ...")
        df = compute_features()
        df.to_csv(feat_file, index=False)

    plot_indicators(df)
    wear = df[df["wear_class"].isin(WEAR_CLASSES)].reset_index(drop=True)

    report("Indicator medians per speed and class:")
    report(df.groupby(["rpm", "wear_class"])[FEATURES].median().round(2).to_string())
    n_takes = wear.groupby(["rpm", "wear_class"])["take_id"].nunique().unstack()[WEAR_CLASSES]
    report(f"\nFiles used: {len(wear)} (fresh/moderate/broken). Original takes per class:\n{n_takes.to_string()}")

    # Main experiment: all 6 indicators, train at one speed, test at the other.
    report("\n=== All 6 indicators, balanced accuracy (chance = 0.33) ===")
    acc_a, pred_a, model_a = cross_speed(wear, 520, 635, FEATURES)
    acc_b, pred_b, _ = cross_speed(wear, 635, 520, FEATURES)
    report(f"  train 520 -> test 635 : {acc_a:.2f}   <- the main question")
    report(f"  train 635 -> test 520 : {acc_b:.2f}")
    report(f"  same speed, new takes : 520 = {same_speed_by_take(wear, 520, FEATURES):.2f}, "
           f"635 = {same_speed_by_take(wear, 635, FEATURES):.2f}   (upper reference)")

    report("\nModel weights trained at 520 rpm (on standardised indicators; + means 'pushes towards this class'):")
    coef = pd.DataFrame(model_a[-1].coef_, index=model_a[-1].classes_, columns=FEATURES).loc[WEAR_CLASSES]
    report(coef.round(2).to_string())

    # Which indicators transfer on their own?
    report("\n=== One indicator at a time, balanced accuracy ===")
    rows = []
    for f in FEATURES:
        rows.append({"indicator": f,
                     "520->635": cross_speed(wear, 520, 635, [f])[0],
                     "635->520": cross_speed(wear, 635, 520, [f])[0],
                     "same speed 520": same_speed_by_take(wear, 520, [f]),
                     "same speed 635": same_speed_by_take(wear, 635, [f])})
    report(pd.DataFrame(rows).set_index("indicator").round(2).to_string())

    # The simplest physical question: fresh vs broken tooth (moderate left out). Chance = 0.50.
    fb = wear[wear["wear_class"].isin(["fresh", "broken"])]
    report("\n=== Fresh vs broken only, balanced accuracy (chance = 0.50) ===")
    for name, feats in [("all 6 indicators", FEATURES), ("tooth minus revolution only", ["order4_minus_1_db"])]:
        report(f"  {name:28s} 520->635 {cross_speed(fb, 520, 635, feats)[0]:.2f}   "
               f"635->520 {cross_speed(fb, 635, 520, feats)[0]:.2f}   "
               f"same speed 520 {same_speed_by_take(fb, 520, feats):.2f}   "
               f"635 {same_speed_by_take(fb, 635, feats):.2f}")

    # Confusion matrices for the two cross-speed directions.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (title, truth, pred, acc) in zip(axes, [
            ("train 520 -> test 635", wear.loc[wear["rpm"] == 635, "wear_class"], pred_a, acc_a),
            ("train 635 -> test 520", wear.loc[wear["rpm"] == 520, "wear_class"], pred_b, acc_b)]):
        cm = confusion_matrix(truth, pred, labels=WEAR_CLASSES, normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=10)
        ax.set_xticks(range(3), WEAR_CLASSES); ax.set_yticks(range(3), WEAR_CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{title}   (balanced accuracy {acc:.2f})")
        ax.grid(False)
    fig.savefig(OUT_DIR / "2_confusion_matrices.png")
    plt.close(fig)

    (OUT_DIR / "results.txt").write_text("\n".join(results), encoding="utf-8")
    print(f"\nSaved results and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
