"""Inspect the segmented milling sound dataset (no model training).

Reads every .wav under SegmentedData/, builds an inventory table, prints
summaries and saves waveform + mel-spectrogram plots for a few random files.
Original files are only read, never changed.

Outputs (all in analysis/ next to SegmentedData/):
    inventory.csv   one row per file
    summary.txt     the same summaries that are printed to the terminal
    plots/          waveform + mel spectrogram figures

Run from the project folder:
    .venv\\Scripts\\python.exe scripts\\inspect_dataset.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # save figures to files, no window needed
import matplotlib.pyplot as plt

# pathlib handles the non-ASCII "Masaüstü" in the path; we never build paths as plain strings.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "SegmentedData"
OUT_DIR = ROOT / "analysis"
PLOT_DIR = OUT_DIR / "plots"

# Use the shared project figure style if available.
sys.path.insert(0, str(ROOT))
try:
    from toolwear.plotting import use_project_style
    use_project_style()
except ImportError:
    pass

CLASSES = ["base", "fresh", "moderate", "broken"]
CLIP_LEVEL = 0.99        # |sample| at or above 99 % of full scale counts as (near) clipping
SEED = 42                # fixed seed so the same plot files are picked every run
FILES_PER_GROUP = 2      # random files plotted per rpm and class
N_MELS = 128
N_FFT = 2048
HOP = 512

# File name pattern: RPM + speed + (BASE or one tool letter) + sample number, e.g. RPM520A0012.wav
NAME_RE = re.compile(r"^RPM(?P<rpm>\d+)(?P<tool>BASE|[A-Z])(?P<sample>\d+)\.wav$", re.IGNORECASE)
# Class folder pattern: class word + speed, e.g. FRESH520
FOLDER_RE = re.compile(r"^(?P<cls>[A-Za-z]+)(?P<rpm>\d+)$")

# Everything printed by report() is also collected here and written to summary.txt.
summary_lines = []


def report(text=""):
    print(text)
    summary_lines.append(str(text))


def parse_file(path):
    """Return the metadata we can get from the file name and folder names."""
    row = {"path": str(path.relative_to(ROOT)), "file": path.name}
    problems = []

    m = NAME_RE.match(path.name)
    if m:
        row["rpm_name"] = int(m["rpm"])
        row["tool"] = m["tool"].upper()
        row["sample"] = int(m["sample"])
    else:
        row["rpm_name"], row["tool"], row["sample"] = None, None, None
        problems.append("file name does not match RPM<speed><tool><number>.wav")

    f = FOLDER_RE.match(path.parent.name)
    if f and f["cls"].lower() in CLASSES:
        row["wear_class"] = f["cls"].lower()
        row["rpm_folder"] = int(f["rpm"])
    else:
        row["wear_class"], row["rpm_folder"] = None, None
        problems.append(f"unknown class folder '{path.parent.name}'")

    # The speed folder one level up (520/ or 635/) should agree as well.
    row["rpm_top_folder"] = int(path.parent.parent.name) if path.parent.parent.name.isdigit() else None

    # Cross-checks between file name and folders.
    rpms = {row["rpm_name"], row["rpm_folder"], row["rpm_top_folder"]} - {None}
    if len(rpms) > 1:
        problems.append(f"rpm disagrees: name={row['rpm_name']}, folder={row['rpm_folder']}, top folder={row['rpm_top_folder']}")
    if row["tool"] is not None and row["wear_class"] is not None:
        if (row["tool"] == "BASE") != (row["wear_class"] == "base"):
            problems.append(f"name says tool '{row['tool']}' but folder says '{row['wear_class']}'")

    row["rpm"] = row["rpm_name"] if row["rpm_name"] is not None else row["rpm_folder"]
    row["name_folder_problem"] = "; ".join(problems)
    return row


def read_audio_stats(path):
    """Read the audio and return basic signal properties."""
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)  # shape (frames, channels), full scale = 1.0
    peak = float(np.max(np.abs(x)))
    stats = {
        "sr": sr,
        "frames": x.shape[0],
        "duration_s": x.shape[0] / sr,
        "channels": x.shape[1],
        "subtype": info.subtype,
        "peak": peak,
        "peak_dbfs": 20 * np.log10(peak) if peak > 0 else -np.inf,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "n_clipped": int(np.sum(np.abs(x) >= CLIP_LEVEL)),
    }
    # For stereo files: are the two channels really different, or a copied mono signal?
    if x.shape[1] == 2:
        left, right = x[:, 0], x[:, 1]
        stats["channels_identical"] = bool(np.array_equal(left, right))
        stats["lr_correlation"] = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else np.nan
    else:
        stats["channels_identical"], stats["lr_correlation"] = None, np.nan
    return stats


def build_inventory():
    files = sorted(DATA_DIR.rglob("*.wav"))
    rows = []
    for i, path in enumerate(files, 1):
        row = parse_file(path)
        try:
            row.update(read_audio_stats(path))
            row["read_error"] = ""
        except Exception as e:  # e.g. an OneDrive online-only file that cannot be downloaded
            row["read_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if i % 200 == 0:
            print(f"  read {i}/{len(files)} files")
    df = pd.DataFrame(rows)
    df["wear_class"] = pd.Categorical(df["wear_class"], categories=CLASSES, ordered=True)
    return df


def print_summaries(df):
    ok = df[df["read_error"] == ""]
    pd.set_option("display.width", 120)

    report("=" * 70)
    report(f"Files found: {len(df)}   readable: {len(ok)}   unreadable: {len(df) - len(ok)}")
    for _, r in df[df["read_error"] != ""].iterrows():
        report(f"  CANNOT READ {r['path']}: {r['read_error']}")

    report("\nName / folder problems:")
    bad = df[df["name_folder_problem"] != ""]
    if bad.empty:
        report("  none - every file name agrees with its folders")
    for _, r in bad.iterrows():
        report(f"  {r['path']}: {r['name_folder_problem']}")

    report("\n1) Number of files per rpm and class")
    report(pd.crosstab(df["rpm"], df["wear_class"], margins=True, dropna=False).to_string())

    report("\n2) Number of DIFFERENT tool letters per rpm and class (= physical tools, if one letter = one tool)")
    n_tools = df.groupby(["rpm", "wear_class"], observed=False)["tool"].nunique().unstack()
    report(n_tools.to_string())
    letters = df.groupby(["rpm", "wear_class"], observed=True)["tool"].agg(lambda s: ", ".join(sorted(s.unique()))).unstack()
    report("   Tool letters in each group:")
    report(letters.to_string())

    report("\n3) Number of recordings per tool letter")
    per_tool = df.groupby(["rpm", "tool"]).agg(
        wear_class=("wear_class", lambda s: ", ".join(sorted(s.astype(str).unique()))),
        n_files=("file", "size"),
        first_sample=("sample", "min"),
        last_sample=("sample", "max"),
    )
    report(per_tool.to_string())
    # Gaps in the sample numbering may mean missing files.
    for (rpm, tool), g in df.groupby(["rpm", "tool"]):
        expected = set(range(int(g["sample"].min()), int(g["sample"].max()) + 1))
        missing = sorted(expected - set(g["sample"]))
        if missing:
            report(f"   RPM{rpm} {tool}: missing sample numbers {missing}")

    report("\n4) Does any tool letter appear in more than one wear class?")
    classes_per_tool = df.groupby(["rpm", "tool"])["wear_class"].nunique()
    shared = classes_per_tool[classes_per_tool > 1]
    if shared.empty:
        report("   Within each rpm: no - every tool letter belongs to exactly one class.")
    else:
        report("   !!! WARNING: these tool letters appear in more than one class:")
        report(shared.to_string())
    across_rpm = df.groupby("tool")["rpm"].nunique()
    across_rpm = across_rpm[across_rpm > 1].index.tolist()
    if across_rpm:
        report(f"   Note: letters {across_rpm} are used at both speeds. The file names cannot tell us")
        report("   whether e.g. 'A' at 520 rpm and 'A' at 635 rpm is the same physical tool.")
        for tool in across_rpm:
            cls = df.loc[df["tool"] == tool, "wear_class"].astype(str).unique()
            report(f"     {tool}: class at both speeds = {sorted(cls)}")

    report("\n5) Audio format")
    report("   Sampling rate (Hz):  " + ok["sr"].value_counts().to_dict().__repr__())
    report("   Channels:            " + ok["channels"].value_counts().to_dict().__repr__())
    report("   Sample format:       " + ok["subtype"].value_counts().to_dict().__repr__())
    report("   Duration (s):")
    report(ok["duration_s"].describe().round(3).to_string())

    majority = {c: ok[c].mode()[0] for c in ["sr", "channels", "frames", "subtype"]}
    report(f"   Majority format: {majority['sr']} Hz, {majority['channels']} channel(s), "
           f"{majority['frames']} frames = {majority['frames'] / majority['sr']:.2f} s, {majority['subtype']}")
    odd = ok[(ok["sr"] != majority["sr"]) | (ok["channels"] != majority["channels"]) | (ok["subtype"] != majority["subtype"])]
    report(f"   Files with a different sampling rate / channel count / format: {len(odd)}")
    for _, r in odd.iterrows():
        report(f"     {r['file']}: {r['sr']} Hz, {r['channels']} ch, {r['subtype']}")

    short = ok[ok["frames"] != majority["frames"]].sort_values(["rpm", "tool", "sample"])
    report(f"   Files with a different length than {majority['frames'] / majority['sr']:.0f} s: {len(short)}")
    report(short.groupby(["rpm", "wear_class"], observed=False).size().unstack().to_string())
    report("   Their durations (s):")
    report(short["duration_s"].describe().round(2).to_string())
    report("   Short files (file: seconds):")
    for (rpm, tool), g in short.groupby(["rpm", "tool"]):
        items = ", ".join(f"{s}:{d:.1f}" for s, d in zip(g["sample"], g["duration_s"]))
        report(f"     RPM{rpm} {tool}: {items}")

    report("\n6) Stereo channels")
    if (ok["channels"] == 2).any():
        st = ok[ok["channels"] == 2]
        report(f"   Files where left and right are bit-identical: {int(st['channels_identical'].sum())} of {len(st)}")
        report("   Left/right correlation per class:")
        report(st.groupby(["rpm", "wear_class"], observed=False)["lr_correlation"].describe()[["min", "50%", "max"]].round(3).to_string())

    report("\n7) Level and clipping")
    report(f"   Peak amplitude (full scale = 1.0) and RMS per group:")
    lvl = ok.groupby(["rpm", "wear_class"], observed=False).agg(
        peak_min=("peak", "min"), peak_median=("peak", "median"), peak_max=("peak", "max"),
        rms_median=("rms", "median"))
    report(lvl.round(4).to_string())
    clipped = ok[ok["peak"] >= CLIP_LEVEL]
    report(f"   Files reaching >= {CLIP_LEVEL} of full scale: {len(clipped)}")
    for _, r in clipped.iterrows():
        report(f"     {r['file']}: peak {r['peak']:.4f}, {r['n_clipped']} samples >= {CLIP_LEVEL}")
    # Many files with exactly the same peak would hint that the audio was peak-normalised per file.
    top_peak = ok["peak"].round(4).value_counts().head(3)
    report(f"   Most common peak values (a spike here would suggest per-file normalisation): {top_peak.to_dict()}")


def make_plots(df):
    """Waveform + mel spectrogram for 2 random files per rpm and class, all on the same scales."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ok = df[(df["read_error"] == "") & df["wear_class"].notna()]
    rng = np.random.default_rng(SEED)

    # Pick the files first.
    picks = []
    for rpm in sorted(ok["rpm"].unique()):
        for cls in CLASSES:
            group = ok[(ok["rpm"] == rpm) & (ok["wear_class"] == cls)]
            idx = rng.choice(len(group), size=min(FILES_PER_GROUP, len(group)), replace=False)
            picks += [group.iloc[i] for i in sorted(idx)]

    # Compute all spectrograms first, so we can use one shared colour scale and waveform y-axis.
    # Only the first (left) channel is plotted.
    items = []
    for r in picks:
        x, sr = sf.read(ROOT / r["path"], dtype="float32", always_2d=True)
        y = x[:, 0]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=0, fmax=sr / 2)
        mel_db = librosa.power_to_db(mel, ref=1.0)  # dB relative to full scale, NOT per-file max, so files are comparable
        items.append((r, y, sr, mel_db))

    vmax = max(np.max(m) for *_, m in items)
    vmin = vmax - 80                                  # show an 80 dB range below the loudest bin
    ylim = max(np.max(np.abs(y)) for _, y, _, _ in items) * 1.05
    fmax = min(sr for _, _, sr, _ in items) / 2       # same frequency range on every plot

    for r, y, sr, mel_db in items:
        fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 2]})
        t = np.arange(len(y)) / sr
        ax_w.plot(t, y, linewidth=0.4)
        ax_w.set_ylim(-ylim, ylim)
        ax_w.set_ylabel("amplitude (full scale = 1)")
        ax_w.set_title(f"{r['file']}   rpm {r['rpm']}, class {r['wear_class']}, tool {r['tool']}   (left channel)")
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                                       fmin=0, fmax=fmax, vmin=vmin, vmax=vmax, cmap="magma", ax=ax_s)
        ax_s.grid(False)  # grid lines would hide spectrogram detail
        ax_s.set_ylabel("frequency (Hz, mel scale)")
        ax_s.set_xlabel("time (s)")
        ax_s.set_xlim(0, 10)  # same time axis even for shorter files
        fig.colorbar(img, ax=[ax_w, ax_s], format="%+.0f dB", label="dB (same fixed reference for all files)")
        fig.savefig(PLOT_DIR / f"{r['rpm']}_{r['wear_class']}_{Path(r['file']).stem}.png")
        plt.close(fig)

    # One overview page with all spectrograms side by side for quick comparison.
    rpms = sorted(ok["rpm"].unique())
    fig, axes = plt.subplots(len(rpms) * FILES_PER_GROUP, len(CLASSES),
                             figsize=(4 * len(CLASSES), 2.6 * len(rpms) * FILES_PER_GROUP),
                             sharex=True, sharey=True, squeeze=False)
    seen = {}  # how many files of this rpm/class are already drawn -> which row to use
    for r, y, sr, mel_db in items:
        k = seen.get((r["rpm"], r["wear_class"]), 0)
        seen[(r["rpm"], r["wear_class"])] = k + 1
        ax = axes[rpms.index(r["rpm"]) * FILES_PER_GROUP + k, CLASSES.index(r["wear_class"])]
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="mel",
                                       fmin=0, fmax=fmax, vmin=vmin, vmax=vmax, cmap="magma", ax=ax)
        ax.set_title(f"{r['rpm']} rpm / {r['wear_class']} / {r['file']}", fontsize=8)
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.colorbar(img, ax=axes, format="%+.0f dB", label="dB (same fixed reference for all files)", shrink=0.6)
    fig.savefig(PLOT_DIR / "overview_all_classes.png")
    plt.close(fig)

    report(f"\nPlots: {len(items)} files plotted (seed {SEED}), colour scale {vmin:.0f} to {vmax:.0f} dB, "
           f"0-{fmax:.0f} Hz, waveform y-axis +/-{ylim:.3f}. Saved to {PLOT_DIR.relative_to(ROOT)}")


def main():
    if not DATA_DIR.is_dir():
        sys.exit(f"Data folder not found: {DATA_DIR}")
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Reading .wav files under {DATA_DIR} ...")
    df = build_inventory()
    df.to_csv(OUT_DIR / "inventory.csv", index=False, encoding="utf-8")
    print(f"Saved {OUT_DIR / 'inventory.csv'}")

    print_summaries(df)
    make_plots(df)

    (OUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
