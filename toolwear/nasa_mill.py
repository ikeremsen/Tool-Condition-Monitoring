"""
Loader and data-quality checks for the NASA / UC Berkeley milling dataset (mill.mat).

The file is a MATLAB 1x167 struct array: one element per run (one face-milling pass).
This module only *reads and checks* the data. It makes no cleaning decisions; those
are made explicitly in the notebooks so they stay visible.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

SCALAR_FIELDS = ["case", "run", "VB", "time", "DOC", "feed", "material"]
SIGNAL_FIELDS = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

# What each field means, from the dataset README (Goebel & Agogino).
FIELD_DESCRIPTIONS = {
    "case": "Experiment number (1-16), one insert set per case",
    "run": "Counter of runs (passes) within a case",
    "VB": "Flank wear [mm], measured with a microscope, not after every run",
    "time": "Cumulative cutting time within the case [min]",
    "DOC": "Depth of cut [mm]",
    "feed": "Feed [mm/rev]",
    "material": "Workpiece material: 1 = cast iron, 2 = steel",
    "smcAC": "Spindle motor current, AC, unprocessed",
    "smcDC": "Spindle motor current, DC portion",
    "vib_table": "Table vibration, RMS envelope",
    "vib_spindle": "Spindle vibration, RMS envelope",
    "AE_table": "Acoustic emission at table, RMS envelope",
    "AE_spindle": "Acoustic emission at spindle, RMS envelope",
}

MATERIAL_NAMES = {1: "cast iron", 2: "steel"}

# README Table 2: case -> (depth of cut [mm], feed [mm/rev], material code)
README_CONDITIONS = {
    1: (1.5, 0.5, 1),   2: (0.75, 0.5, 1),  3: (0.75, 0.25, 1), 4: (1.5, 0.25, 1),
    5: (1.5, 0.5, 2),   6: (1.5, 0.25, 2),  7: (0.75, 0.25, 2), 8: (0.75, 0.5, 2),
    9: (1.5, 0.5, 1),  10: (1.5, 0.25, 1), 11: (0.75, 0.25, 1), 12: (0.75, 0.5, 1),
    13: (0.75, 0.25, 2), 14: (0.75, 0.5, 2), 15: (1.5, 0.25, 2), 16: (1.5, 0.5, 2),
}

# Machine settings from the README (Matsuura MC-510V, 70 mm face mill, 6 KC710 inserts)
SPINDLE_RPM = 826
CUTTER_DIAMETER_MM = 70
WORKPIECE_LENGTH_MM = 483
FS_README_HZ = 250        # stated sampling rate; see notebook 01, it does not fully add up
DAQ_FULL_SCALE_V = 10.0   # +-10 V input range; smcDC tops out at 9.995 V


def _scalar(x):
    """MATLAB scalars arrive as 1x1 arrays; an empty entry would be 0x0."""
    x = np.asarray(x)
    return float(x.ravel()[0]) if x.size else np.nan


def load_mill(mat_path):
    """
    Read mill.mat.

    Returns
    -------
    meta : DataFrame, one row per record, with the scalar fields plus 'record'
           (position in the file) and 'n_samples'.
    signals : list of dicts, signals[i][channel] -> 1-D numpy array for record i.
    """
    records = sio.loadmat(Path(mat_path))["mill"].ravel()

    meta = pd.DataFrame([{f: _scalar(r[f]) for f in SCALAR_FIELDS} for r in records])
    meta.insert(0, "record", np.arange(len(meta)))
    for col in ["case", "run", "material"]:
        meta[col] = meta[col].astype(int)

    signals = [{ch: r[ch].ravel().astype(float) for ch in SIGNAL_FIELDS} for r in records]
    meta["n_samples"] = [s["smcAC"].size for s in signals]
    return meta, signals


def field_storage(mat_path):
    """How each field is stored across all records: 'dtype shape' -> count."""
    records = sio.loadmat(Path(mat_path))["mill"].ravel()
    out = {}
    for name in records.dtype.names:
        combos = pd.Series([f"{r[name].dtype} {r[name].shape}" for r in records]).value_counts()
        out[name] = ", ".join(f"{k} x{v}" for k, v in combos.items())
    return pd.Series(out, name="storage (dtype shape x count)")


def find_corrupt_records(meta, signals, limit_volts=100.0, expected_length=9000):
    """
    Flag records that cannot be physical measurements.

    The sensors were conditioned to about +-5 V and the DAQ reads +-10 V, so any
    |value| above `limit_volts` is corrupted data. A wrong length is also flagged.
    """
    rows = []
    for i, sig in enumerate(signals):
        bad_idx = np.unique(np.concatenate(
            [np.flatnonzero(np.abs(x) > limit_volts) for x in sig.values()]))
        n = sig["smcAC"].size
        if n != expected_length or bad_idx.size:
            rows.append({
                "record": i, "case": meta.case[i], "run": meta.run[i], "VB": meta.VB[i],
                "n_samples": n, "n_bad_samples": bad_idx.size,
                "bad_from": bad_idx.min() if bad_idx.size else None,
                "bad_to": bad_idx.max() if bad_idx.size else None,
            })
    return pd.DataFrame(rows)


def clipped_fraction(x, rail=9.99):
    """Share of samples sitting at the top of the DAQ input range (clipped)."""
    return float(np.mean(x >= rail))


def in_cut_window(smcDC, level=0.5):
    """
    Rough start/end sample of the actual cut, from the DC spindle current.

    The current jumps when the cutter engages the workpiece and falls when it
    leaves, so we take the first and last sample above `level` of its range.
    """
    thr = smcDC.min() + level * (smcDC.max() - smcDC.min())
    above = np.flatnonzero(smcDC > thr)
    return int(above[0]), int(above[-1])
