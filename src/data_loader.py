import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Direkte Download-URLs von rruff.info
RRUFF_URLS = {
    "fair_oriented":       "https://rruff.info/zipped_data_files/raman/LR%20Raman%20Data%2C%20FAIR%2C%20Oriented.zip",
    "fair_unoriented":     "https://rruff.info/zipped_data_files/raman/LR%20Raman%20Data%2C%20FAIR%2C%20Unoriented.zip",
    "excellent_oriented":  "https://rruff.info/zipped_data_files/raman/LR%20Raman%20Data%2C%20EXCELLENT%2C%20Oriented.zip",
}


def load_rruff(split: str = "fair_oriented", save_csv: bool = True):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prüfen ob Daten schon lokal vorhanden
    csv_path = PROCESSED_DATA_DIR / f"rruff_{split}.csv"
    if csv_path.exists():
        print(f"[data_loader] Lade aus lokalem CSV: {csv_path}")
        return load_from_csv(split)

    # ZIP herunterladen
    if split not in RRUFF_URLS:
        raise ValueError(f"Unbekannter Split '{split}'. Wähle: {list(RRUFF_URLS.keys())}")

    zip_path = RAW_DATA_DIR / f"{split}.zip"

    if not zip_path.exists():
        url = RRUFF_URLS[split]
        print(f"[data_loader] Lade herunter: {url}")
        print("[data_loader] Das dauert beim ersten Mal 2–5 Minuten ...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[data_loader] Gespeichert: {zip_path}")

    # ZIP einlesen und Spektren parsen
    print("[data_loader] Lese Spektren aus ZIP ...")
    wavenumbers_list, intensities_list, labels_list = _parse_rruff_zip(zip_path)

    wavenumbers = np.array(wavenumbers_list, dtype=object)
    intensities = np.array(intensities_list, dtype=object)
    labels      = np.array(labels_list)

    print(f"[data_loader] Fertig: {len(labels)} Spektren, {len(set(labels))} Mineralklassen")

    if save_csv:
        _save_to_csv(wavenumbers, intensities, labels, split)

    return wavenumbers, intensities, labels


def _parse_rruff_zip(zip_path: Path):
    """
    Öffnet das ZIP und liest alle .txt Dateien.

    Jede .txt Datei = ein Spektrum.
    Header-Zeilen beginnen mit ## und werden übersprungen.
    Datenzeilen haben das Format: wavenumber, intensity
    """
    wavenumbers_list = []
    intensities_list = []
    labels_list      = []
    skipped          = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        txt_files = [f for f in zf.namelist() if f.endswith('.txt')]
        print(f"[data_loader] {len(txt_files)} Dateien im ZIP gefunden")

        for filename in txt_files:
            try:
                with zf.open(filename) as f:
                    content = f.read().decode('utf-8', errors='ignore')

                mineral_name, wn, inten = _parse_single_spectrum(content, filename)

                if wn is not None and len(wn) > 10:
                    wavenumbers_list.append(wn)
                    intensities_list.append(inten)
                    labels_list.append(mineral_name)
                else:
                    skipped += 1

            except Exception:
                skipped += 1
                continue

    print(f"[data_loader] {len(labels_list)} Spektren geladen, {skipped} übersprungen")
    return wavenumbers_list, intensities_list, labels_list


def _parse_single_spectrum(content: str, filename: str):
    """
    Parst eine einzelne RRUFF .txt Datei.

    Gibt zurück: (mineralname, wavenumbers_array, intensities_array)
    oder (filename, None, None) bei Fehler.
    """
    mineral_name = Path(filename).stem  # Dateiname als Fallback
    wavenumbers  = []
    intensities  = []

    for line in content.splitlines():
        line = line.strip()

        # Leerzeilen überspringen
        if not line:
            continue

        # Header-Zeilen: Mineralname extrahieren
        if line.startswith("##NAMES="):
            name = line.replace("##NAMES=", "").strip()
            if name:
                # Nimm nur den ersten Namen falls mehrere angegeben
                mineral_name = name.split(",")[0].strip()
            continue

        # Alle anderen Header-Zeilen überspringen
        if line.startswith("#"):
            continue

        # Datenzeile parsen: "wavenumber, intensity"
        # Robustes Parsen: split by Komma, nimm erste zwei Werte
        parts = line.replace("\t", ",").split(",")
        if len(parts) >= 2:
            try:
                wn    = float(parts[0].strip())
                inten = float(parts[1].strip())
                wavenumbers.append(wn)
                intensities.append(inten)
            except ValueError:
                continue  # Zeile die sich nicht parsen lässt überspringen

    if len(wavenumbers) > 10:
        return mineral_name, np.array(wavenumbers), np.array(intensities)
    else:
        return mineral_name, None, None


def load_from_csv(split: str = "fair_oriented"):
    """
    Lädt bereits gespeicherten Datensatz aus data/processed/ (schneller als Re-Download).
    Wird ab Notebook 02 verwendet.
    """
    path = PROCESSED_DATA_DIR / f"rruff_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {path}\n"
            f"Bitte zuerst load_rruff('{split}') aufrufen."
        )
    print(f"[data_loader] Lade von CSV: {path}")
    df = pd.read_csv(path)

    label_col    = "mineral"
    feature_cols = [c for c in df.columns if c != label_col]

    intensities  = df[feature_cols].values
    labels       = df[label_col].values
    wavenumbers  = np.array([float(c) for c in feature_cols])

    return wavenumbers, intensities, labels


def _save_to_csv(wavenumbers, intensities, labels, split: str):
    """
    Speichert Daten als CSV.
    Da Spektren unterschiedliche Längen haben können,
    resampled wir auf eine gemeinsame Achse.
    """
    from scipy.interpolate import interp1d

    # Gemeinsame Wellenzahl-Achse: 200–3500 cm⁻¹, 1000 Punkte
    target_wn = np.linspace(200, 3500, 1000)
    resampled  = []

    for wn, inten in zip(wavenumbers, intensities):
        try:
            # Nur Bereich der tatsächlich vorhanden ist interpolieren
            mask = (wn >= target_wn[0]) & (wn <= target_wn[-1])
            if mask.sum() < 10:
                continue
            f        = interp1d(wn[mask], inten[mask],
                                kind='linear', bounds_error=False, fill_value=0.0)
            resampled.append(f(target_wn))
        except Exception:
            continue

    resampled   = np.array(resampled)
    # Labels auf gleiche Länge bringen (falls einige übersprungen wurden)
    valid_count = len(resampled)
    labels_cut  = labels[:valid_count]

    col_names = [str(round(w, 2)) for w in target_wn]
    df        = pd.DataFrame(resampled, columns=col_names)
    df.insert(0, "mineral", labels_cut)

    out_path = PROCESSED_DATA_DIR / f"rruff_{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"[data_loader] CSV gespeichert: {out_path}  ({len(resampled)} Spektren)")