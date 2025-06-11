#zero-shot  
import os

# Redirect cache directories to scratch before importing whisper/evaluate
os.environ["HF_HOME"] = "/scratch/s5910587/cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/s5910587/cache"
os.environ["TORCH_HOME"] = "/scratch/s5910587/cache/torch"
os.environ["XDG_CACHE_HOME"] = "/scratch/s5910587/cache"

import csv
import torch
import whisper
from tqdm import tqdm
from evaluate import load
import re

BASE_INPUT_DIR = "/scratch/s5910587/data_asd/segments"
GT_DIR = "/scratch/s5910587/data_asd/GT"
BASE_OUTPUT_DIR = "/scratch/s5910587/whisper_zs/transcription_ZS_large-v2"

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

wer_metric = load("wer")
# pre-trained Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large-v2", device=device)
# Supported multilingual models: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"

wer_summary = []
all_preds, all_refs = [], []
group_predictions = []
group_references = []

def transcribe_and_evaluate(subfolder):
    input_folder = os.path.join(BASE_INPUT_DIR, subfolder)
    gt_csv = os.path.join(GT_DIR, f"{subfolder}.csv")
    out_csv = os.path.join(BASE_OUTPUT_DIR, f"{subfolder}.csv")

    if not os.path.exists(input_folder) or not os.path.exists(gt_csv):
        print(f"❌ Missing folder or ground truth: {subfolder}")
        return

    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # Load ground truth CSV
    gt = {}
    with open(gt_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            gt[row[0]] = row[1]

    predictions, references = [], []

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcription"])

        for wav_file in tqdm(sorted(os.listdir(input_folder)), desc=f"🔍 {subfolder}"):
            if not wav_file.endswith(".wav"):
                continue
            if wav_file not in gt:
                # 如果没有对应的 ground truth，则删除该音频文件
                os.remove(os.path.join(input_folder, wav_file))
                continue
            wav_path = os.path.join(input_folder, wav_file)
            result = model.transcribe(wav_path, language="nl", fp16=torch.cuda.is_available())  # nl is the language code for Dutch
            writer.writerow([wav_file, result["text"]])
            if wav_file in gt:
                predictions.append(result["text"])
                references.append(gt[wav_file])

    if predictions:
        predictions_norm = [normalize(p) for p in predictions]
        references_norm = [normalize(r) for r in references]
        wer = wer_metric.compute(predictions=predictions_norm, references=references_norm)
        wer_summary.append([subfolder, f"{wer:.4f}"])
        all_preds.extend(predictions)
        all_refs.extend(references)
    return predictions, references

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for sub in sorted(os.listdir(BASE_INPUT_DIR)):
    if os.path.isdir(os.path.join(BASE_INPUT_DIR, sub)):
        predictions, references = transcribe_and_evaluate(sub)
        if predictions and references:
            group_predictions.append(predictions)
            group_references.append(references)
            if len(group_predictions) == 20:
                flat_preds = sum(group_predictions, [])
                flat_refs = sum(group_references, [])
                group_wer = wer_metric.compute(predictions=[normalize(p) for p in flat_preds],
                                               references=[normalize(r) for r in flat_refs])
                group_id = len(wer_summary) // 20
                wer_summary.append([f"GROUP_{group_id}", f"{group_wer:.4f}"])
                group_predictions, group_references = [], []

if group_predictions:
    flat_preds = sum(group_predictions, [])
    flat_refs = sum(group_references, [])
    group_wer = wer_metric.compute(predictions=[normalize(p) for p in flat_preds],
                                   references=[normalize(r) for r in flat_refs])
    group_id = len(wer_summary) // 20
    wer_summary.append([f"GROUP_{group_id}", f"{group_wer:.4f}"])

#write WER summary to CSV
summary_file = os.path.join(BASE_OUTPUT_DIR, "wer_summary.csv")
with open(summary_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["speaker", "WER"])
    writer.writerows(wer_summary)
    if all_preds:
        all_preds_norm = [normalize(p) for p in all_preds]
        all_refs_norm = [normalize(r) for r in all_refs]
        total_wer = wer_metric.compute(predictions=all_preds_norm, references=all_refs_norm)
        writer.writerow(["OVERALL", f"{total_wer:.4f}"])

print(f"WER summary saved to {summary_file}")
