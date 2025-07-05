"""
파일명 : GAN_Restore.py
해당 파일 목적 : GAN.py 에서 학습한 모델을 바탕으로 손실 데이터 생성 및 복원
"""

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# GAN.py에서 만든 모델 불러오기
loaded_generator = load_model('GANModel_Robut.h5')

# 데이터 불러오고 전처리
data = pd.read_csv('D:\hdj\OneDrive - 공주대학교\대학\연구실\공부\태양열 복원 GAN\Preprocessing_Data\Processed_IVT_Data.csv')
data['Date'] = pd.to_datetime(data['Date'])
train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=False)
train_dates = train_data.pop('Date')
valid_dates = valid_data.pop('Date')

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
valid_scaled = scaler.transform(valid_data)

# 손실 데이터 생성
num_samples, num_features = valid_scaled.shape
missing_data = valid_scaled.copy()

missing_percentage = 0.3  #Data Missing 30%
corrupted_percentage = 0.3  #Data Corrupted 30%
inconsistent_percentage = 0.4  #Data Inconsistent 40%

num_missing = int(missing_percentage * num_samples)
num_corrupted = int(corrupted_percentage * num_samples)
num_inconsistent = num_samples - (num_missing + num_corrupted)

# 손실 유형에 따라서 복원률을 각각 측정할것이기 때문에 index 생성 및 저장
missing_indices = []
corrupted_indices = []
inconsistent_indices = []

# Data Missing 생성 부분
missing_sample_indices = np.random.choice(num_samples, num_missing, replace=False)
missing_feature_indices = np.random.choice(num_features, num_missing, replace=True)
for sample_idx, feature_idx in zip(missing_sample_indices, missing_feature_indices):
    missing_data[sample_idx, feature_idx] = np.nan
    missing_indices.append((sample_idx, feature_idx))

# Data Corrupted 생성 부분
corrupted_sample_indices = np.random.choice(num_samples, num_corrupted, replace=False)
corrupted_feature_indices = np.random.choice(num_features, num_corrupted, replace=True)
for sample_idx, feature_idx in zip(corrupted_sample_indices, corrupted_feature_indices):
    missing_data[sample_idx, feature_idx] *= -1
    corrupted_indices.append((sample_idx, feature_idx))

# Data Inconsistent 생성 부분
inconsistent_sample_indices = np.random.choice(num_samples, num_inconsistent, replace=False)
inconsistent_feature_indices = np.random.choice(num_features, num_inconsistent, replace=True)
for sample_idx, feature_idx in zip(inconsistent_sample_indices, inconsistent_feature_indices):
    missing_data[sample_idx, feature_idx] = np.random.uniform(-1, 1)
    inconsistent_indices.append((sample_idx, feature_idx))

# 데이터 복원
latent_dim = 100

def restore_missing_data_with_loaded_model(missing_data, model, altered_indices):
    restored_data = missing_data.copy()
    total_altered = len(altered_indices)
    for idx, (sample_idx, feature_idx) in enumerate(altered_indices):
        noise = np.random.normal(0, 1, (1, latent_dim))
        generated_data = model.predict(noise)
        restored_data[sample_idx, feature_idx] = generated_data.flatten()[feature_idx]
        if (idx + 1) % 100 == 0 or idx + 1 == total_altered:  #복원 진행도 표시
            print(f"Restored {idx + 1}/{total_altered} samples.")
    return restored_data

#복원과 시각화
restored_data_missing = restore_missing_data_with_loaded_model(missing_data, loaded_generator, missing_indices)
restored_data_corrupted = restore_missing_data_with_loaded_model(missing_data, loaded_generator, corrupted_indices)
restored_data_inconsistent = restore_missing_data_with_loaded_model(missing_data, loaded_generator, inconsistent_indices)

#수치적 평가 진행. MSE, MAE
def evaluate_restoration(original, restored, missing_indices):
    mse = mean_squared_error(original[~np.isnan(missing_data)], restored[~np.isnan(missing_data)])
    r2 = r2_score(original[~np.isnan(missing_data)], restored[~np.isnan(missing_data)])
    mae = mean_absolute_error(original[~np.isnan(missing_data)], restored[~np.isnan(missing_data)])
    return mse, r2, mae

mse_missing, r2_missing, mae_missing = evaluate_restoration(valid_scaled, restored_data_missing, missing_indices)
mse_corrupted, r2_corrupted, mae_corrupted = evaluate_restoration(valid_scaled, restored_data_corrupted, corrupted_indices)
mse_inconsistent, r2_inconsistent, mae_inconsistent = evaluate_restoration(valid_scaled, restored_data_inconsistent, inconsistent_indices)

#결과 출력
print("Data Missing")
print(f"Mean Squared Error: {mse_missing}")
print(f"R-Squared: {r2_missing}")
print(f"Mean Absolute Error: {mae_missing}")

print("\nData Corrupted")
print(f"Mean Squared Error: {mse_corrupted}")
print(f"R-Squared: {r2_corrupted}")
print(f"Mean Absolute Error: {mae_corrupted}")

print("\nData Inconsistent")
print(f"Mean Squared Error: {mse_inconsistent}")
print(f"R-Squared: {r2_inconsistent}")
print(f"Mean Absolute Error: {mae_inconsistent}")

# 복원률 계산 및 시각화
def calculate_and_visualize_restoration_rate(original, restored, altered_indices, title, subplot):
    # altered_indices가 비어있는 경우를 처리
    if not altered_indices:
        print(f"\n{title}")
        print("No data altered for this category.")
        return

    missing_row_indices = np.unique([idx[0] for idx in altered_indices])
    original_rows_with_missing = original[missing_row_indices]
    restored_rows = restored[missing_row_indices]

    # NaN 값을 무시하고 MAE 계산. NaN 있으경우 오류 발생
    mae_per_row = np.nanmean(np.abs(original_rows_with_missing - restored_rows), axis=1)
    restoration_rate = 1 - mae_per_row
    average_restoration_rate = np.nanmean(restoration_rate) * 100

    print(f"\n{title}")
    print(f"Average Restoration Rate: {average_restoration_rate:.2f}%")

    subplot.hist(restoration_rate * 100, bins=20, edgecolor='black', color='skyblue')
    subplot.set_xlabel("Restoration Rate (%)")
    subplot.set_ylabel("Frequency")
    subplot.set_title(f"{title}")

fig, axs = plt.subplots(1, 3, figsize=(21, 6))

calculate_and_visualize_restoration_rate(valid_scaled, restored_data_missing, missing_indices, "Data Missing", axs[0])
calculate_and_visualize_restoration_rate(valid_scaled, restored_data_corrupted, corrupted_indices, "Data Corrupted", axs[1])
calculate_and_visualize_restoration_rate(valid_scaled, restored_data_inconsistent, inconsistent_indices, "Data Inconsistent", axs[2])

plt.tight_layout()
plt.show()
