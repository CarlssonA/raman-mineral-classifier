"""
preprocessing.py
================
Preprocessing-Pipeline für Raman-Spektren.

Schritte:
1. Resampling        — alle Spektren auf dieselbe Wellenzahl-Achse
2. Baseline-Korrektur — Fluoreszenz-Hintergrund entfernen (SNIP)
3. Min-Max-Normierung — Intensitäten auf [0, 1] skalieren
"""

import numpy as np
from scipy.interpolate import interp1d


def resample_spectrum(wn, intensity, target_wn):
    mask = (wn >= target_wn[0]) & (wn <= target_wn[-1])
    if mask.sum() < 5:
        return np.zeros(len(target_wn))
    f = interp1d(wn[mask], intensity[mask],
                 kind='linear', bounds_error=False, fill_value=0.0)
    return f(target_wn)


def resample_spectra(wavenumbers, intensities, target_wn):
    n = intensities.shape[0]
    result = np.zeros((n, len(target_wn)))
    for i in range(n):
        wn = wavenumbers[i] if wavenumbers.ndim == 2 else wavenumbers
        result[i] = resample_spectrum(wn, intensities[i], target_wn)
    return result


def correct_baseline_snip(intensities, max_half_window=40):
    try:
        from pybaselines import Baseline
        fitter = Baseline()
        corrected = np.zeros_like(intensities)
        for i in range(len(intensities)):
            baseline, _ = fitter.snip(intensities[i], max_half_window=max_half_window)
            corrected[i] = np.clip(intensities[i] - baseline, 0, None)
        return corrected
    except ImportError:
        return np.clip(intensities - intensities.min(axis=1, keepdims=True), 0, None)


def normalize_minmax(intensities):
    mins = intensities.min(axis=1, keepdims=True)
    maxs = intensities.max(axis=1, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    return (intensities - mins) / ranges


def full_pipeline(wavenumbers, intensities, target_wn=None, baseline_method='snip'):
    """
    Vollständige Preprocessing-Pipeline.
    Schritte: Resampling → Baseline-Korrektur → Min-Max-Normierung

    Rückgabe: (target_wn, X_processed)
    """
    if target_wn is None:
        target_wn = np.linspace(200, 3500, 1000)

    X = resample_spectra(wavenumbers, intensities, target_wn)

    if baseline_method == 'snip':
        X = correct_baseline_snip(X)

    X = normalize_minmax(X)

    return target_wn, X