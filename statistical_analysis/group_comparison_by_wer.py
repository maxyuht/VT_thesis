import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import os

# === Step 1: Load data ===
csv_path = "/scratch/s5910587/features_analysis/merged_summary_with_wer_cer.csv"  # 替换为你的本地路径
df = pd.read_csv(csv_path)

# === Step 2: 设置 WER 的高低分组 ===
# 你可以调整分位数比例，例如 25%
high_wer_thresh = df['WER_sentence'].quantile(0.75)
low_wer_thresh = df['WER_sentence'].quantile(0.25)

# 创建组别列
df['WER_group'] = df['WER_sentence'].apply(
    lambda x: 'Low-WER' if x <= low_wer_thresh else ('High-WER' if x >= high_wer_thresh else 'Mid')
)

# 只保留两组（去掉中间那50%）
df_filtered = df[df['WER_group'].isin(['Low-WER', 'High-WER'])]

# === Step 3: 分析的特征 ===
features = [
    'mean_pitch_Hz', 'pitch_range_Hz', 'pitch_SD_Hz',
    'mean_intensity_dB', 'intensity_SD_dB',
    'shimmer_local', 'voiced_ratio', 'speech_rate_voiced_frames_per_sec',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_flatness'
]

# === Step 4: 统计检验并可视化 ===
results = []
output_dir = "/scratch/s5910587/statistical_analysis"
os.makedirs(output_dir, exist_ok=True)

for feat in features:
    group1 = df_filtered[df_filtered['WER_group'] == 'Low-WER'][feat].dropna()
    group2 = df_filtered[df_filtered['WER_group'] == 'High-WER'][feat].dropna()

    # 检查是否正态分布（这里只是示意，实际需用shapiro检验）
    # 使用非参数检验更稳妥：Mann–Whitney U 检验
    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

    results.append({'Feature': feat, 'Mann-Whitney U p-value': p})

    # 可视化
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df_filtered, x='WER_group', y=feat, palette='Set2')
    plt.title(f'{feat} by WER Group\n(p = {p:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{feat}_violin.png"), dpi=300)
    plt.close()

# === Step 5: 保存统计结果 ===
pd.DataFrame(results).to_csv(os.path.join(output_dir, "mann_whitney_results.csv"), index=False)
print("分析完成，图表和结果已保存。")
