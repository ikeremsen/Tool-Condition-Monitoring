"""Step 2 inspection: look for the milling physics in the sound (no model training).

Milling is periodic. Each cutting edge hits the workpiece once per revolution, so
    spindle frequency        fs = rpm / 60   (nominal; measured values in SPINDLE_HZ)
    tooth-passing frequency  fz = flutes * rpm / 60
Hypotheses from the literature:
    - wear changes the strength of the tooth-passing harmonics (more rubbing force)
    - a broken tooth makes the pattern repeat once per revolution instead of once
      per tooth, so energy moves from tooth harmonics to spindle harmonics
    - anything also present in BASE (machine only) is not about the tool

Figures (analysis/spectral/):
    1_low_freq_spectrum_<rpm>.png   average spectrum 0-250 Hz, fine resolution
    2_envelope_spectrum_<rpm>.png   how often the cutting noise "pulses" (0-100 Hz)
    3_minus_base_<rpm>.png          each class minus the machine-only sound, 20 Hz-24 kHz
    4_tooth_vs_spindle.png          per-file test of the broken-tooth hypothesis
Numbers: analysis/spectral/per_file_features.csv, analysis/spectral/summary.txt

Only full 10 s files are used (short tail pieces would change the frequency resolution).
Only the left channel is used: both channels are identical in this dataset.

Run from the project folder:
    .venv\\Scripts\\python.exe scripts\\spectral_inspection.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "analysis" / "inventory.csv"   # made by inspect_dataset.py
OUT_DIR = ROOT / "analysis" / "spectral"

sys.path.insert(0, str(ROOT))
from toolwear.plotting import use_project_style, SERIES, TEXT_MUTED, GRID
use_project_style()

FLUTES = 4                 # assumed number of cutting edges (user's observation of the 16 mm end mill)
# Spindle speed measured from the data, not the nominal rpm/60 (8.67 and 10.58 Hz). The envelope of the
# cutting noise has peaks at 33.2 Hz and 40.0 Hz, exactly 4 x these values, so the cutter turns ~4-5 %
# slower than the nominal setting. Peaks at the nominal values also exist, already in BASE (machine only).
SPINDLE_HZ = {520: 8.30, 635: 10.01}
CLASSES = ["base", "fresh", "moderate", "broken"]
COLORS = {"base": TEXT_MUTED, "fresh": SERIES[0], "moderate": SERIES[1], "broken": SERIES[2]}

FINE_NPERSEG = 2 ** 17     # 131072 samples = 2.7 s windows -> 0.37 Hz resolution for the low-frequency spectrum
WIDE_NPERSEG = 4096        # 11.7 Hz resolution, enough for the 20 Hz-24 kHz overview
# Band whose envelope (loudness over time) we analyse. 10-20 kHz because figure 3 shows that is
# where cutting is 5-15 dB louder than the machine alone; below ~8 kHz BASE is as loud or louder.
ENV_BAND = (10000, 20000)  # Hz
ENV_FS = 1000              # envelope is resampled to 1 kHz, enough to see pulses up to ~100 Hz
ENV_NPERSEG = 8192         # 8.2 s windows at 1 kHz -> 0.12 Hz resolution


def load_left(path):
    x, sr = sf.read(ROOT / path, dtype="float32", always_2d=True)
    return x[:, 0], sr


def envelope(x, sr):
    """Band-pass the cutting noise, take its amplitude envelope, and resample it to ENV_FS."""
    sos = signal.butter(4, ENV_BAND, btype="bandpass", fs=sr, output="sos")
    band = signal.sosfiltfilt(sos, x)
    env = np.abs(signal.hilbert(band))
    env = env - env.mean()                              # remove the average loudness, keep the fluctuation
    return signal.resample_poly(env, ENV_FS, sr)        # includes an anti-aliasing filter


def band_power(f, p, centre, rel_width=0.02):
    """Power in a +/- 2 % window around a frequency (allows for small speed changes during a cut)."""
    m = (f >= centre * (1 - rel_width)) & (f <= centre * (1 + rel_width))
    return p[m].sum()


def compute_spectra(files):
    """One row of spectra per file. Returns frequency axes and a dict of 2-D arrays (files x freq)."""
    fine, wide, env = [], [], []
    for i, path in enumerate(files.path, 1):
        x, sr = load_left(path)
        f_fine, p = signal.welch(x, sr, nperseg=FINE_NPERSEG); fine.append(p)
        f_wide, p = signal.welch(x, sr, nperseg=WIDE_NPERSEG); wide.append(p)
        f_env, p = signal.welch(envelope(x, sr), ENV_FS, nperseg=ENV_NPERSEG); env.append(p)
        if i % 200 == 0:
            print(f"  {i}/{len(files)} files")
    return (f_fine, np.array(fine)), (f_wide, np.array(wide)), (f_env, np.array(env))


def db(p):
    return 10 * np.log10(p + 1e-20)


def harmonic_lines(ax, rpm, fmax):
    """Dotted grey = spindle harmonics, solid dark = tooth-passing harmonics (every FLUTES-th spindle harmonic)."""
    fs = SPINDLE_HZ[rpm]
    for k in range(1, int(fmax / fs) + 1):
        tooth = k % FLUTES == 0
        ax.axvline(k * fs, color="#333333" if tooth else "#999999",
                   linestyle="-" if tooth else ":", linewidth=0.9 if tooth else 0.7, zorder=0)


def plot_class_spectra(f, spectra, groups, rpm, fmax, title, ylabel, fname, fmin=0):
    """Median spectrum per class with the middle 50 % of files shaded."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    m = (f >= fmin) & (f <= fmax)
    for cls in CLASSES:
        d = db(spectra[groups == cls][:, m])
        q25, med, q75 = np.percentile(d, [25, 50, 75], axis=0)
        ax.fill_between(f[m], q25, q75, color=COLORS[cls], alpha=0.15, linewidth=0)
        ax.plot(f[m], med, color=COLORS[cls], linewidth=1.0, label=f"{cls} (n={int((groups == cls).sum())})")
    harmonic_lines(ax, rpm, fmax)
    ax.set_xlim(fmin, fmax)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {rpm} rpm   (dotted: spindle {SPINDLE_HZ[rpm]:.2f} Hz harmonics, "
                 f"solid: tooth passing {FLUTES * SPINDLE_HZ[rpm]:.1f} Hz harmonics for {FLUTES} flutes)")
    ax.grid(False)
    ax.legend(loc="upper right")
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)


def plot_minus_base(f, spectra, groups, rpm, fname):
    """How much louder (dB) each cutting class is than the machine alone, per frequency."""
    m = (f >= 20) & (f <= f.max())
    base = np.median(db(spectra[groups == "base"][:, m]), axis=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for cls in CLASSES[1:]:
        ax.plot(f[m], np.median(db(spectra[groups == cls][:, m]), axis=0) - base,
                color=COLORS[cls], linewidth=1.0, label=cls)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(20, f.max())
    ax.set_xlabel("frequency (Hz, log scale)")
    ax.set_ylabel("class minus BASE (dB)")
    ax.set_title(f"Sound added by cutting, compared with the machine alone - {rpm} rpm   (above 0 = louder than BASE)")
    ax.legend(loc="upper left")
    fig.savefig(OUT_DIR / fname)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(INVENTORY)
    full = inv[(inv["read_error"].isna()) & (inv["frames"] == inv["frames"].mode()[0])].reset_index(drop=True)
    print(f"Using {len(full)} full-length files (of {len(inv)})")

    lines = []
    rows = []
    for rpm in sorted(full["rpm"].unique()):
        files = full[full["rpm"] == rpm].reset_index(drop=True)
        groups = files["wear_class"].to_numpy()
        print(f"{rpm} rpm: computing spectra for {len(files)} files ...")
        (f_fine, fine), (f_wide, wide), (f_env, env) = compute_spectra(files)

        plot_class_spectra(f_fine, fine, groups, rpm, 250, "Average sound spectrum, low frequencies",
                           "power (dB)", f"1_low_freq_spectrum_{rpm}.png", fmin=2)
        plot_class_spectra(f_env, env, groups, rpm, 100, f"Envelope spectrum of the {ENV_BAND[0] // 1000}-"
                           f"{ENV_BAND[1] // 1000} kHz noise", "envelope power (dB)",
                           f"2_envelope_spectrum_{rpm}.png", fmin=2)
        plot_minus_base(f_wide, wide, groups, rpm, f"3_minus_base_{rpm}.png")

        # Where are the strongest envelope pulses really? (actual speed can differ from nominal)
        lines.append(f"\n{rpm} rpm: nominal spindle {rpm / 60:.2f} Hz, measured {SPINDLE_HZ[rpm]:.2f} Hz, "
                     f"tooth passing {FLUTES * SPINDLE_HZ[rpm]:.2f} Hz")
        for cls in CLASSES:
            med = np.median(env[groups == cls], axis=0)
            m = (f_env > 3) & (f_env < 100)
            top = np.sort(f_env[m][np.argsort(med[m])[::-1][:5]])
            lines.append(f"  {cls:9s} strongest envelope peaks (Hz): {np.round(top, 2).tolist()}")

        # Per-file broken-tooth test: tooth-passing pulses vs once-per-revolution pulses in the envelope.
        fs, fz = SPINDLE_HZ[rpm], FLUTES * SPINDLE_HZ[rpm]
        for i, r in files.iterrows():
            spindle = band_power(f_env, env[i], fs)
            tooth = band_power(f_env, env[i], fz)
            rows.append({"file": r["file"], "rpm": rpm, "wear_class": r["wear_class"], "sample": r["sample"],
                         "env_spindle_db": db(spindle), "env_tooth_db": db(tooth),
                         "tooth_minus_spindle_db": db(tooth) - db(spindle)})

    feats = pd.DataFrame(rows)
    feats.to_csv(OUT_DIR / "per_file_features.csv", index=False)

    # Figure 4: one box per class and rpm.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, rpm in zip(axes, sorted(feats["rpm"].unique())):
        d = feats[feats["rpm"] == rpm]
        data = [d.loc[d["wear_class"] == c, "tooth_minus_spindle_db"] for c in CLASSES]
        bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, widths=0.6, showfliers=True,
                        flierprops={"markersize": 2, "markeredgecolor": TEXT_MUTED})
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(0.5)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(f"{rpm} rpm")
    axes[0].set_ylabel("tooth-passing minus spindle pulse power (dB)")
    fig.suptitle("Broken-tooth test: above 0 = pulses once per tooth dominate, below 0 = once per revolution dominate",
                 fontsize=10)
    fig.savefig(OUT_DIR / "4_tooth_vs_spindle.png")
    plt.close(fig)

    lines.append("\nTooth-passing minus spindle pulse power (dB), median per class:")
    lines.append(feats.groupby(["rpm", "wear_class"])["tooth_minus_spindle_db"].median()
                 .unstack()[CLASSES].round(1).to_string())
    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(f"Saved figures and numbers to {OUT_DIR}")


if __name__ == "__main__":
    main()
