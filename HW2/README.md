# HW2 – Automatic Speech Recognition

## Overview

In this assignment you will build and evaluate a simple Automatic Speech Recognition
(ASR) pipeline, going from raw audio to transcribed text.

## Learning Objectives

- Understand the end-to-end ASR pipeline
- Work with pre-trained acoustic models (e.g. `openai/whisper` or `facebook/wav2vec2`)
- Evaluate transcription quality using Word Error Rate (WER)
- Experiment with fine-tuning or decoding parameters

## Deliverables

| File | Description |
|------|-------------|
| `HW2.ipynb` | Completed Jupyter notebook with all code, outputs, and answers |
| `report.pdf` *(optional)* | Short written report (if required by your instructor) |

## Submission Deadline

**TBD** – check the course portal for the exact date.

## Getting Started

```bash
# Install dependencies (once)
pip install -r ../requirements.txt

# Launch the notebook
jupyter notebook HW2.ipynb
```

## Grading

| Task | Points |
|------|--------|
| Audio loading & pre-processing | 15 |
| Running a pre-trained ASR model | 30 |
| WER evaluation | 25 |
| Experiments & analysis | 30 |
| **Total** | **100** |
