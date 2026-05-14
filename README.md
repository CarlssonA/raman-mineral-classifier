# Raman Mineral Classifier

Automatic mineral identification from Raman spectra using machine learning — trained on the [RRUFF database](https://rruff.info).

---

## Project Overview

Raman spectroscopy produces a characteristic "fingerprint" spectrum for each mineral. This project trains a Random Forest classifier to identify minerals from their Raman spectra, achieving a **weighted F1-score of 0.95** across 100+ mineral classes.

A Streamlit web app allows users to upload their own spectrum (CSV) and receive a mineral prediction with confidence score.

---

## Results

| Model | Dataset | Classes | Weighted F1 |
|---|---|---|---|
| Random Forest (200 trees) | RRUFF excellent_oriented | 100+ | 0.95 |

- Training/test split: 80/20, stratified
- Features: preprocessed spectra (SNIP baseline correction + Min-Max normalization, 1000 points)
- Class imbalance handled via `class_weight='balanced'`

---

## Project Structure

```
raman-mineral-classifier/
├── app/
│   └── streamlit_app.py       # Interactive web demo
├── data/
│   ├── raw/                   # Downloaded RRUFF ZIP files
│   └── processed/             # Resampled spectra (CSV + NPY)
├── models/
│   ├── rf_model.pkl           # Trained Random Forest
│   ├── label_encoder.pkl      # Class name mapping
│   └── target_wavenumbers.npy
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Preprocessing pipeline
│   └── 03_modeling.ipynb      # Model training & evaluation
├── src/
│   ├── data_loader.py         # RRUFF data download & parsing
│   ├── preprocessing.py       # Baseline correction, normalization
│   └── features.py            # Peak detection & feature extraction
└── requirements.txt
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app/streamlit_app.py
```

To retrain the model, run the notebooks in order: `01_eda` → `02_preprocessing` → `03_modeling`.

---

## Data

The [RRUFF database](https://rruff.info) (University of Arizona) provides reference Raman spectra for minerals worldwide. This project uses the `excellent_oriented` split (~1,500+ spectra, 100+ mineral classes).

Data is downloaded automatically on first run via `src/data_loader.py`. Alternatively, download the ZIP manually from [rruff.info/zipped_data_files/raman/](https://rruff.info/zipped_data_files/raman/) and place it in `data/raw/`.

---

## Tech Stack

Python · scikit-learn · SciPy · pybaselines · NumPy · pandas · Streamlit · Plotly

---

## A Note on AI Assistance

This project was developed with support from Claude (Anthropic) as part of a portfolio-building process. I want to be transparent about how and where AI was used:

**Work I carried out independently:**
The data analysis and modeling notebooks (EDA, preprocessing, model training and evaluation) reflect my own understanding of the underlying concepts — Raman spectroscopy, ML pipelines, cross-validation, and evaluation metrics. I worked through each step myself, using AI assistance primarily for debugging errors and clarifying concepts, much like one would use Stack Overflow or documentation.

**Where AI assistance was more substantial:**
The Streamlit app, the `preprocessing.py` and `features.py` modules, and the overall project scaffolding were developed with more direct AI support. I chose this approach deliberately: in modern data science workflows, leveraging AI to build visualization tools and pipeline infrastructure is itself a practical skill. Being able to define what you need, evaluate whether the output is correct, and integrate it into a larger project is meaningful technical work — even when the code is AI-assisted.

**Why I'm disclosing this:**
Transparency matters more than the appearance of having done everything from scratch. I believe the ability to work effectively with AI tools — knowing when to use them, how to verify their output, and how to adapt them to your specific problem — is an increasingly important competency in data science.

---

## Author

**Carlsson Arlt** — M.Sc. Nanoscience  
[GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/)
