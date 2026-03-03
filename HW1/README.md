# HW1 – Speech Feature Extraction

## Overview

In this assignment you will load raw audio files, compute standard speech features,
and perform basic analysis and visualisation.

## Learning Objectives

- Read and pre-process audio data with `librosa` / `scipy`
- Compute time-domain and frequency-domain features
  - Waveform, energy, zero-crossing rate
  - Short-Time Fourier Transform (STFT) / Spectrogram
  - Mel-Frequency Cepstral Coefficients (MFCCs)
- Visualise features and draw conclusions from them

## Deliverables

| File | Description |
|------|-------------|
| `HW1.ipynb` | Completed Jupyter notebook with all code, outputs, and answers |
| `report.pdf` *(optional)* | Short written report (if required by your instructor) |

## Submission Deadline

**TBD** – check the course portal for the exact date.

## Getting Started

```bash
# Install dependencies (once)
pip install -r ../requirements.txt

# Launch the notebook
jupyter notebook HW1.ipynb
```

## Grading

| Task | Points |
|------|--------|
| Audio loading & pre-processing | 20 |
| Spectrogram computation & plot | 25 |
| MFCC computation & plot | 25 |
| Analysis & written answers | 30 |
| **Total** | **100** |
