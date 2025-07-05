import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np  # numpy import 추가

# GPU 사용 여부 확인
print("GPU 사용 가능 여부:", tf.config.list_physical_devices('GPU'))

# 1단계: 데이터셋 로드 및 전처리
file_path = 'code/combined_processed_IVT_data.csv'
data = pd.read_csv(file_path)

# 날짜 열을 인덱스로 설정
data.set_index('Date', inplace=True)  # 실제 날짜 컬럼 이름으로 교체

# 데이터 정규화 (0~1 사이의 값으로 스케일링)
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

# 2단계: MPTCP를 이용한 데이터 전송 시뮬레이션
def transmit_data(data, loss_probability=0.1):
    transmitted_data = {}
    for date, row in data.iterrows():
        transmitted_row = []
        for value in row:
            if random.random() > loss_probability:
                transmitted_row.append(value)
            else:
                transmitted_row.append(None)  # 패킷 손실 시뮬레이션
        transmitted_data[date] = transmitted_row
    return transmitted_data

transmitted_data = transmit_data(data_scaled)  # 데이터를 전송 시뮬레이션

# 3단계: GAN 모델 구축 및 학습
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 데이터셋에 따라 입력과 출력 크기를 지정
input_dim = 100  # 노이즈 벡터 크기
output_dim = data.shape[1]  # 데이터 차원 (열의 개수)

generator = build_generator(input_dim, output_dim)
discriminator = build_discriminator(output_dim)
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(input_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=32):
    for epoch in tqdm(range(epochs), desc="Training GAN"):  # tqdm으로 진행 상황 출력
        # 판별자 학습
        noise = tf.random.normal([batch_size, input_dim])
        generated_data = generator(noise)
        real_data = data.sample(batch_size)
        combined_data = tf.concat([real_data, generated_data], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        discriminator.train_on_batch(combined_data, labels)

        # 생성자 학습
        noise = tf.random.normal([batch_size, input_dim])
        misleading_labels = tf.ones((batch_size, 1))
        gan.train_on_batch(noise, misleading_labels)

# GAN 학습
train_gan(gan, generator, discriminator, pd.DataFrame(data_scaled), epochs=1000, batch_size=32)  # 학습 에포크와 배치 크기는 조정 가능

# 4단계: 패킷 복구 시뮬레이션
def recover_lost_packets(transmitted_data, generator):
    recovered_data = []
    for date, row in transmitted_data.items():
        recovered_row = []
        for value in row:
            if value is None:
                noise = tf.random.normal([1, input_dim])
                with tf.device('/CPU:0'):  # GPU를 사용하는 동안 출력 억제
                    recovered_value = generator(noise, training=False)
                recovered_row.append(recovered_value[0])
            else:
                recovered_row.append(value)
        recovered_data.append(recovered_row)
    return np.array(recovered_data)

recovered_data = recover_lost_packets(transmitted_data, generator)

# 복구된 데이터를 원래 스케일로 복원
recovered_data = scaler.inverse_transform(recovered_data)

# 5단계: 성능 평가
def evaluate_recovery(original_data, recovered_data):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(original_data, recovered_data)
    return loss

# 원본 데이터와 복구된 데이터 비교
recovery_loss = evaluate_recovery(data.values, recovered_data)
print(f"Recovery MSE: {recovery_loss}")
