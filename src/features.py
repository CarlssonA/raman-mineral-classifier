"""
features.py
===========
Feature-Extraktion aus vorverarbeiteten Raman-Spektren.

Für den Random Forest: Peak-basierte Features
Für den CNN: rohe normierte Spektren (kein Feature-Engineering nötig)
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis


def detect_peaks(spectrum, height=0.05, prominence=0.03, distance=10):
    """
    Findet Peaks in einem normierten Spektrum.

    Parameter
    ---------
    spectrum   : 1D np.ndarray (normiert auf [0,1])
    height     : Mindesthöhe eines Peaks
    prominence : Mindestprominenz (wie stark ragt der Peak heraus?)
    distance   : Mindestabstand zwischen zwei Peaks (in Indizes)

    Rückgabe
    --------
    peaks : Indizes der gefundenen Peaks
    props : Eigenschaften der Peaks (Höhen, Prominenzen)
    """
    peaks, props = find_peaks(spectrum, height=height,
                              prominence=prominence, distance=distance)
    return peaks, props


def extract_peak_features(X, wavenumbers, n_bins=30):
    """
    Extrahiert Peak-basierte Features für den Random Forest.

    Features pro Spektrum:
    - n_bins Histogramm-Bins: wie viele Peaks fallen in welchen Wellenzahl-Bereich?
    - Mean, Std, Skewness, Kurtosis der Intensitäten
    - Anzahl der Peaks

    Gesamt: n_bins + 5 Features

    Parameter
    ---------
    X          : np.ndarray shape (n_samples, n_points) — normierte Spektren
    wavenumbers: 1D np.ndarray — Wellenzahl-Achse
    n_bins     : Anzahl der Histogramm-Bins

    Rückgabe
    --------
    np.ndarray shape (n_samples, n_bins + 5)
    """
    n_samples = X.shape[0]
    n_features = n_bins + 5
    features = np.zeros((n_samples, n_features))

    bin_edges = np.linspace(wavenumbers[0], wavenumbers[-1], n_bins + 1)

    for i, spectrum in enumerate(X):
        # Peak-Detection
        peaks, _ = detect_peaks(spectrum)
        peak_wn = wavenumbers[peaks] if len(peaks) > 0 else np.array([])

        # Histogramm: Anzahl Peaks pro Wellenzahl-Bereich
        if len(peak_wn) > 0:
            hist, _ = np.histogram(peak_wn, bins=bin_edges)
        else:
            hist = np.zeros(n_bins)

        features[i, :n_bins] = hist

        # Spektrale Statistiken
        features[i, n_bins]     = np.mean(spectrum)
        features[i, n_bins + 1] = np.std(spectrum)
        features[i, n_bins + 2] = float(skew(spectrum))
        features[i, n_bins + 3] = float(kurtosis(spectrum))
        features[i, n_bins + 4] = len(peaks)

    return features


def prepare_cnn_input(X):
    """
    Bereitet Spektren für den 1D-CNN vor.
    Fügt eine Kanal-Dimension hinzu: (n_samples, n_points) → (n_samples, n_points, 1)

    Parameter
    ---------
    X : np.ndarray shape (n_samples, n_points)

    Rückgabe
    --------
    np.ndarray shape (n_samples, n_points, 1)
    """
    return X[..., np.newaxis]