import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv("/scratch/s5910587/features_analysis/merged_summary_with_wer_cer.csv")

# 定义用于分析的特征列
feature_cols = [
    'mean_pitch_Hz', 'pitch_range_Hz', 'pitch_SD_Hz',
    'mean_intensity_dB', 'intensity_SD_dB',
    'shimmer_local', 'voiced_ratio',
    'speech_rate_voiced_frames_per_sec',
    'spectral_centroid', 'spectral_bandwidth',
    'spectral_rolloff', 'spectral_flatness'
]

# 目标变量列
target_cols = ['WER_sentence', 'CER_sentence']

# 提取用于相关性分析的子数据框
correlation_df = df[feature_cols + target_cols]

# 计算 Spearman 相关性矩阵
correlation_matrix = correlation_df.corr(method='spearman')

# 绘制热力图并保存为图片（适用于无图形界面的远程服务器）
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix[target_cols].T, annot=True, cmap='coolwarm', center=0)
plt.title("Spearman Correlation: Acoustic Features vs WER / CER")
plt.tight_layout()
plt.savefig("/scratch/s5910587/statistical_analysis/spearman_heatmap.png")  # 保存路径
plt.close()  # 关闭图像，防止内存问题