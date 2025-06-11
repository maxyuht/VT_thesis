import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# === Step 1: 读取数据 ===
csv_path = "/scratch/s5910587/features_analysis/merged_summary_with_wer_cer.csv"  # 修改为你的CSV文件路径
df = pd.read_csv(csv_path)

# === Step 2: 选择特征与目标变量 ===
feature_cols = [
    'mean_pitch_Hz', 'pitch_range_Hz', 'pitch_SD_Hz',
    'mean_intensity_dB', 'intensity_SD_dB',
    'shimmer_local', 'voiced_ratio',
    'speech_rate_voiced_frames_per_sec',
    'spectral_centroid', 'spectral_bandwidth',
    'spectral_rolloff', 'spectral_flatness'
]

X = df[feature_cols]
y = df['WER_sentence']

# === Step 3: 添加常数项（intercept）并拟合OLS回归模型 ===
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# === Step 4: 打印回归结果 ===
print(model.summary())

# === Step 5: 提取回归系数和p值，保存为CSV ===
results_df = pd.DataFrame({
    'Feature': model.params.index,
    'Coefficient': model.params.values,
    'P-value': model.pvalues.values
})
results_df.to_csv("/scratch/s5910587/statistical_analysis/multiple_regression_WER_results.csv", index=False)

# === Step 6: 可视化回归系数（不含截距） ===
coef_only = results_df[results_df['Feature'] != 'const']
plt.figure(figsize=(10, 6))
plt.barh(coef_only['Feature'], coef_only['Coefficient'], color='skyblue')
plt.xlabel("Coefficient Value")
plt.title("Regression Coefficients for Predicting WER")
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("/scratch/s5910587/statistical_analysis/multiple_regression_WER_barplot.png", dpi=300)
plt.show()