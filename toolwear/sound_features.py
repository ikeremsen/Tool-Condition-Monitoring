"""
Physical sound indicators for the end-milling audio dataset (Kaggle: cutting-tool-wear-audio-dataset).

Every indicator has a mechanical reason and is made independent of spindle speed where
possible, so a model trained at one speed can be tested at the other:

    hf_energy_db       energy of the 10-20 kHz cutting noise, minus the machine-only (BASE) level  [dB]
    spectral_centroid  "brightness": average frequency of the sound, weighted by power             [kHz]
    hf_kurtosis        spikiness of the 10-20 kHz noise (impacts, chipping); 3 = pure random noise [-]
    order1_db          how strongly the cutting noise pulses once per revolution                   [dB]
    order4_db          how strongly it pulses once per tooth (4 flutes)                            [dB]
    order4_minus_1_db  order4_db - order1_db; drops if one tooth is missing (broken-tooth test)    [dB]

order*_db are peak heights above the envelope's own noise floor, so the overall loudness of a
recording (microphone distance, gain) cancels out.
"""

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, stats

FLUTES = 4
# Spindle speed under cutting load, measured from the envelope peaks in step 2
# (nominal rpm/60 would be 8.67 and 10.58 Hz; the cutter turns ~4-5 % slower).
SPINDLE_HZ = {520: 8.30, 635: 10.01}

HF_BAND = (10000, 20000)   # Hz, where cutting is 5-15 dB louder than the machine alone (step 2, figure 3)
ENV_FS = 1000              # envelope resampled to 1 kHz
ENV_NPERSEG = 8192         # 0.12 Hz resolution in the envelope spectrum
ORDER_WIDTH = 0.02         # search +/- 2 % around each order for the peak

FEATURES = ["hf_energy_db", "spectral_centroid", "hf_kurtosis", "order1_db", "order4_db", "order4_minus_1_db"]


def load_left(path):
    """Read a wav file as float (full scale = 1). Both channels are identical, so keep the left one."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x[:, 0].astype(np.float64), sr


def order_peak_db(f, p, freq, floor):
    """Highest envelope power within +/- ORDER_WIDTH of freq, in dB above the noise floor."""
    m = (f >= freq * (1 - ORDER_WIDTH)) & (f <= freq * (1 + ORDER_WIDTH))
    return 10 * np.log10(p[m].max() / floor)


def raw_features(x, sr, rpm):
    """All indicators for one recording. hf_power is still absolute here; see add_base_reference()."""
    # Whole-spectrum features
    f, p = signal.welch(x, sr, nperseg=4096)
    centroid_khz = np.sum(f * p) / np.sum(p) / 1000

    # The cutting-noise band
    sos = signal.butter(4, HF_BAND, btype="bandpass", fs=sr, output="sos")
    hf = signal.sosfiltfilt(sos, x)
    hf_power = np.mean(hf ** 2)
    kurt = stats.kurtosis(hf, fisher=False)   # 3 for Gaussian noise, higher when there are sharp impacts

    # Envelope (loudness over time) of the cutting noise, and its spectrum
    env = np.abs(signal.hilbert(hf))
    env = signal.resample_poly(env - env.mean(), ENV_FS, sr)
    fe, pe = signal.welch(env, ENV_FS, nperseg=ENV_NPERSEG)
    floor = np.median(pe[(fe >= 2) & (fe <= 100)])     # typical level between the peaks
    fs = SPINDLE_HZ[rpm]
    o1 = order_peak_db(fe, pe, fs, floor)
    o4 = order_peak_db(fe, pe, FLUTES * fs, floor)

    return {"hf_power": hf_power, "spectral_centroid": centroid_khz, "hf_kurtosis": kurt,
            "order1_db": o1, "order4_db": o4, "order4_minus_1_db": o4 - o1}


def add_base_reference(df):
    """hf_energy_db = cutting-noise energy relative to the median BASE (machine only) file at the same speed."""
    df = df.copy()
    base_level = df[df["wear_class"] == "base"].groupby("rpm")["hf_power"].median()
    df["hf_energy_db"] = 10 * np.log10(df["hf_power"] / df["rpm"].map(base_level))
    return df


def assign_takes(inventory):
    """Number the original continuous recordings ("takes").

    The dataset was made by cutting longer recordings into 10 s pieces; the leftover end of each
    recording is a short file. So within one rpm and tool, sorted by sample number, a new take
    starts after every short file. Pieces of the same take must never be split between train and test.
    """
    inv = inventory.sort_values(["rpm", "tool", "sample"]).copy()
    full_len = inv["frames"].mode()[0]
    take_ids = []
    for _, g in inv.groupby(["rpm", "tool"], sort=False):
        take = 0
        for frames in g["frames"]:
            take_ids.append(take)
            if frames != full_len:      # this short piece ends the take
                take += 1
    inv["take"] = take_ids
    inv["take_id"] = inv["rpm"].astype(str) + "_" + inv["tool"] + "_" + inv["take"].astype(str)
    return inv
