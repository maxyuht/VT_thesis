
# README: Dutch-Speaking ASD Child ASR Experiment (Whisper Fine-Tuning)

This repository accompanies the master's thesis *"Improving ASR for Dutch-Speaking Autistic Children through Whisper Fine-Tuning"*. It includes all key steps for model evaluation and training:

1. Zero-shot recognition using Whisper
2. Full-parameter fine-tuning (baseline)
3. LoRA-based fine-tuning
4. Model evaluation and prosody analysis

---

## 📌 Zero-Shot Recognition

Zero-shot recognition was conducted using both `Whisper-large-v2` and `Whisper-medium` models on unadapted speech data.

### ▶ Scripts Used

- `zs_whisper_lgv2.py`: for Whisper-large-v2
- `zs_whisper_m.py`: for Whisper-medium  
Location:  
`./whisper_fine-tuning_asd/whisper_zero_shot/`

> Note: All parameters (model paths, data paths, etc.) are hard-coded into the scripts.

### 📁 Output Directories

- `transcription_ZS_large-v2/`
- `transcription_ZS_m/`  
Located at:  
`./whisper_fine-tuning_asd/whisper_zero_shot/`

---

## 🔧 Whisper Fine-Tuning

### 🧪 Full-Parameter Fine-Tuning (Baseline)

Each experimental group (e.g., ADHD+ASD) has its own Python script and output directory.

- Example script:  
`./whisper_fine-tuning_asd/whisper_ft/adhd_asd/whisper_ft_adhd_asd.py`

- All training configs (model name, paths, hyperparameters) are defined inside the script;
- Output includes trained checkpoints and training logs, stored in group-specific folders.

---

### 🧪 LoRA Fine-Tuning

Parameter-efficient fine-tuning is implemented using HuggingFace's [`peft`](https://github.com/huggingface/peft) library.

- Example script:  
`./whisper_fine-tuning_asd/whisper_ft/asd_only_lora/whisper_ft_asd_lora.py`

- Each LoRA experiment has a dedicated directory for checkpoints and logs;
- The input format is the same as full fine-tuning.

---

## 📈 Evaluation and Prosody Analysis

### 🗂️ Evaluation Output

Sentence-level prediction results (WER, CER) are stored in:

```
./whisper_fine-tuning_asd/result/whisper_eval_sentence/
```

### 🎼 Prosodic Feature Extraction

The following script extracts prosodic descriptors (pitch, intensity, jitter, shimmer, etc.) and generates visualizations:

```
./whisper_fine-tuning_asd/result/features_analysis/prosody_analysis.py
```

### 📊 Statistical Analysis

All scripts for statistical testing and correlation analysis are located in:

```
./whisper_fine-tuning_asd/result/statistical_analysis/
```

These scripts investigate the relationship between prosody/disfluency and ASR performance.

---

## 🧪 Environment Setup

All dependencies are listed in `requirements.txt`.

To set up your Python environment:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Some tasks also require `ffmpeg` and `Praat` for audio processing and annotation alignment.

---


