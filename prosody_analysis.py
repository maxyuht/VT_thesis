import matplotlib.pyplot as plt
def plot_pitch_intensity(df, pause_df, output_path, title=""):
    plt.figure(figsize=(12, 5))
    plt.plot(df["time"], df["pitch_Hz"].ffill(), label="Pitch (Hz)", color="blue")
    plt.plot(df["time"], df["intensity_dB"].ffill(), label="Intensity (dB)", color="orange")
    for _, row in pause_df.iterrows():
        plt.axvspan(row["pause_start"], row["pause_end"], color='gray', alpha=0.3)
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title(f"Pitch & Intensity: {title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
import os
import parselmouth
import numpy as np
import pandas as pd
import librosa

def extract_frame_features(wav_path, frame_step=0.005):
    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch_ac(pitch_floor=75, pitch_ceiling=500, time_step=frame_step)
    pitch_times = pitch.xs()
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    intensity = snd.to_intensity(time_step=frame_step)
    intensity_values = np.interp(pitch_times, intensity.xs(), intensity.values[0])
    voiced_flags = ~np.isnan(pitch_values)
    df = pd.DataFrame({
        "time": pitch_times,
        "pitch_Hz": pitch_values,
        "intensity_dB": intensity_values,
        "voiced": voiced_flags.astype(int)
    })
    return df, snd

def detect_pauses(df, min_pause_duration=0.2):
    pauses = []
    is_in_pause = False
    pause_start = None
    for i in range(len(df)):
        time = df.loc[i, "time"]
        voiced = df.loc[i, "voiced"]
        if voiced == 0 and not is_in_pause:
            is_in_pause = True
            pause_start = time
        elif voiced == 1 and is_in_pause:
            is_in_pause = False
            pause_end = time
            duration = pause_end - pause_start
            if duration >= min_pause_duration:
                pauses.append((pause_start, pause_end, duration))
    if is_in_pause:
        pause_end = df.loc[len(df) - 1, "time"]
        duration = pause_end - pause_start
        if duration >= min_pause_duration:
            pauses.append((pause_start, pause_end, duration))
    pause_df = pd.DataFrame(pauses, columns=["pause_start", "pause_end", "pause_duration"])
    pause_df.index.name = "pause_index"
    return pause_df

def compute_spectral_features(y, sr):
    return {
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        "spectral_rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "spectral_flatness": np.mean(librosa.feature.spectral_flatness(y=y)),
    }

def compute_summary_stats(df, snd, wav_path):
    voiced_df = df[df["voiced"] == 1]
    voiced_pitch = voiced_df["pitch_Hz"].dropna()
    intensity = voiced_df["intensity_dB"].dropna()
    # 替换原来的 point_process 和 jitter/shimmer 调用
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    try:
        jitter_local = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except:
        jitter_local = np.nan
    try:
        shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except:
        shimmer_local = np.nan
    total_duration = snd.get_total_duration()
    voiced_ratio = len(voiced_df) / len(df) if len(df) > 0 else np.nan
    speech_rate = len(voiced_df) / total_duration if total_duration > 0 else np.nan
    y, sr = librosa.load(wav_path, sr=None)
    spectral_feats = compute_spectral_features(y, sr)
    summary = {
        "mean_pitch_Hz": voiced_pitch.mean(),
        "max_pitch_Hz": voiced_pitch.max(),
        "min_pitch_Hz": voiced_pitch.min(),
        "pitch_range_Hz": voiced_pitch.max() - voiced_pitch.min(),
        "pitch_SD_Hz": voiced_pitch.std(),
        "mean_intensity_dB": intensity.mean(),
        "max_intensity_dB": intensity.max(),
        "min_intensity_dB": intensity.min(),
        "intensity_range_dB": intensity.max() - intensity.min(),
        "intensity_SD_dB": intensity.std(),
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
        "voiced_ratio": voiced_ratio,
        "speech_rate_voiced_frames_per_sec": speech_rate,
    }
    summary.update(spectral_feats)
    return pd.DataFrame([summary])

def save_to_excel(df, pause_df, summary_df, output_excel):
    with pd.ExcelWriter(output_excel) as writer:
        df.to_excel(writer, sheet_name='FrameLevel', index=False)
        pause_df.to_excel(writer, sheet_name='Pauses')
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

def batch_process(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            wav_path = os.path.join(input_folder, filename)
            print(f"🔍 Processing {filename} ...")
            try:
                df, snd = extract_frame_features(wav_path)
                pause_df = detect_pauses(df)
                summary_df = compute_summary_stats(df, snd, wav_path)
                basename = os.path.splitext(filename)[0]
                output_excel_path = os.path.join(output_folder, f"{basename}.xlsx")
                save_to_excel(df, pause_df, summary_df, output_excel_path)
                output_plot_path = os.path.join(output_folder, f"{basename}.png")
                plot_pitch_intensity(df, pause_df, output_plot_path, title=basename)
                print(f"✅ Saved: {output_excel_path} and {output_plot_path}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    input_dir = "/scratch/s5910587/data_asd/segs_test/asd38"
    output_dir = "/scratch/s5910587/features_analysis/asd/asd38"
    batch_process(input_dir, output_dir)