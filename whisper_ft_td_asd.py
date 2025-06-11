import os
# === 环境变量设置：缓存重定向到 /scratch/s5910587/cache ===
os.environ["HF_HOME"] = "/scratch/s5910587/cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/s5910587/cache"
os.environ["TORCH_HOME"] = "/scratch/s5910587/cache"
os.environ["XDG_CACHE_HOME"] = "/scratch/s5910587/cache"

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict, Audio
import evaluate
import csv
from tqdm import tqdm
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
import pandas
import matplotlib.pyplot as plt


# === 路径设置 ===
SEGMENTS_TRAIN_DIR = "/scratch/s5910587/data_td_asd/segments"
GT_TRAIN_DIR = "/scratch/s5910587/data_td_asd/GT"
SEGMENTS_VAL_DIR = "/scratch/s5910587/data_asd/segs_val"
GT_VAL_DIR = "/scratch/s5910587/data_asd/GT_val"


# === 特征提取和处理器 ===
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="Dutch", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Dutch", task="transcribe")



# === 固定划分数据加载 ===
def collect_fixed_data():
    """
    从固定的训练和验证目录加载数据，返回 DatasetDict，包含 train/validation。
    """
    import pandas as pd

    def collect_entries_from_dirs(segments_dir, gt_dir):
        entries = []
        speaker_dirs = sorted([d for d in os.listdir(segments_dir) if os.path.isdir(os.path.join(segments_dir, d))])
        for speaker in speaker_dirs:
            speaker_path = os.path.join(segments_dir, speaker)
            gt_csv = os.path.join(gt_dir, f"{speaker}.csv")
            if not os.path.exists(gt_csv):
                continue
            df = pd.read_csv(gt_csv)
            for _, row in df.iterrows():
                fname = row["filename"]
                trans = row["transcription"]
                wav_path = os.path.join(speaker_path, fname)
                if os.path.exists(wav_path):
                    entries.append({
                        "path": wav_path,
                        "filename": fname,
                        "transcription": trans
                    })
        return entries

    train_entries = collect_entries_from_dirs(SEGMENTS_TRAIN_DIR, GT_TRAIN_DIR)
    val_entries = collect_entries_from_dirs(SEGMENTS_VAL_DIR, GT_VAL_DIR)

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_entries).cast_column("path", Audio(sampling_rate=16000)).rename_column("path", "audio"),
        "validation": Dataset.from_list(val_entries).cast_column("path", Audio(sampling_rate=16000)).rename_column("path", "audio")
    })
    ds_dict = ds_dict.remove_columns(["filename"])
    return ds_dict


# 旧的 K-fold 数据加载函数已注释或删除
# def collect_fold_data(fold_id, n_folds=5):
#     ...


# 使用固定划分数据加载
dataset = collect_fixed_data()

print(dataset)


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


dataset["train"] = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names, num_proc=4)
dataset["validation"] = dataset["validation"].map(prepare_dataset, remove_columns=dataset["validation"].column_names, num_proc=4)


# === 数据整理器 ===
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# === 评估指标 ===
metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# === 模型加载与冻结 ===
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

for param in model.parameters():
    param.requires_grad = False

# encoder 最后4层
for i in range(8, 12):
    for param in model.model.encoder.layers[i].parameters():
        param.requires_grad = True

# decoder 全部层
for param in model.model.decoder.parameters():
    param.requires_grad = True

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# === 自定义回调 LoggingCallback ===
class LoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("step\ttrain_loss\teval_loss\teval_wer\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            print("LoggingCallback: logs is None.")
            return
        train_loss = logs.get("loss")
        eval_loss = logs.get("eval_loss")
        eval_wer = logs.get("eval_wer")
        step = state.global_step
        if step is None:
            print("LoggingCallback: state.global_step is None.")
            return
        # Only write log if at least one metric is available
        if train_loss is None and eval_loss is None and eval_wer is None:
            print(f"LoggingCallback: No metrics to log at step {step}.")
            return
        # Write log line with step as integer
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{int(step)}\t{float(train_loss) if train_loss is not None else ''}\t{float(eval_loss) if eval_loss is not None else ''}\t{float(eval_wer) if eval_wer is not None else ''}\n")

        # 读取现有日志文件，绘制曲线图
        try:
            df = pandas.read_csv(self.log_file, sep="\t")
            plt.figure(figsize=(10, 6))
            plt.title("Training Metrics Over Steps")
            plt.xlabel("Step")
            plt.grid(True)

            if 'train_loss' in df.columns and df['train_loss'].notnull().any():
                plt.plot(df['step'], df['train_loss'], label="Train Loss")
            if 'eval_loss' in df.columns and df['eval_loss'].notnull().any():
                plt.plot(df['step'], df['eval_loss'], label="Eval Loss")
            if 'eval_wer' in df.columns and df['eval_wer'].notnull().any():
                plt.plot(df['step'], df['eval_wer'], label="Eval WER")

            plt.legend()
            plt.tight_layout()
            plt.savefig("/scratch/s5910587/whisper_ft/td_asd_90/train_log.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting training log: {e}")


# === 训练参数 ===
training_args = Seq2SeqTrainingArguments(
    output_dir="/scratch/s5910587/whisper_ft/td_asd_90/model",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,  # 降低一点学习率，更稳定
    weight_decay=0.01,   # 防止过拟合

    warmup_steps=5,
    max_steps=90,

    gradient_checkpointing=False,
    fp16=True,

    evaluation_strategy="steps",
    eval_steps=10,
    save_steps=10000,

    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,

    logging_steps=1,
    logging_strategy="steps",
    logging_first_step=True,

    lr_scheduler_type="linear",
    report_to=["tensorboard"],
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


# === 训练器 ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[LoggingCallback("/scratch/s5910587/whisper_ft/td_asd_90/train_log.tsv")], # 只保留日志回调
)


trainer.train()

# === 手动保存模型和 processor 到最终目录（训练完成后） ===
model.save_pretrained("/scratch/s5910587/whisper_ft/td_asd_90/final_model")
processor.save_pretrained("/scratch/s5910587/whisper_ft/td_asd_90/final_processor")

# === 保存最终训练结果 ===
final_metrics = trainer.state.log_history[-1]
import json
with open("/scratch/s5910587/whisper_ft/td_asd_90/final_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2)
