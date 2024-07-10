import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tkinter as tk
from tkinter import scrolledtext
from tkinter import font

# 데이터 읽기
data = pd.read_excel('korean.xlsx')

# 데이터 정제 함수
def clean_text(text):
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", text)
    return text

# 텍스트 정제
comments = data.iloc[:, 0].apply(clean_text)
emotions = data.iloc[:, 1]

# 감정 데이터의 분포 확인
emotion_counts = emotions.value_counts()
plt.figure(figsize=(8, 6))
emotion_counts.plot(kind='bar')
plt.title("Emotion Distribution in the Dataset")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.show()

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(comments, emotions, test_size=0.2, random_state=42)

# 학습 데이터 프레임 생성
train_df = pd.DataFrame({'comment': X_train, 'emotion': y_train})

# 각 감정별로 데이터 나누기
train_df_list = [train_df[train_df['emotion'] == emotion] for emotion in train_df['emotion'].unique()]

# 가장 적은 감정 데이터 수에 맞춰 샘플링
min_count = min(len(df_emotion) for df_emotion in train_df_list)
train_df_resampled = pd.concat([resample(df_emotion, replace=True, n_samples=min_count, random_state=42) for df_emotion in train_df_list])

# 샘플링된 데이터로 학습 데이터 재분리
X_train_resampled = train_df_resampled['comment']
y_train_resampled = train_df_resampled['emotion']

# 텍스트 토큰화
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train_resampled)

X_train_seq = tokenizer.texts_to_sequences(X_train_resampled)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# 패딩 추가
maxlen = 250
X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)

# 감정 인코딩
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_resampled)

# 테스트 데이터 인코딩 (라벨 인코더에 존재하지 않는 라벨 제거)
y_test = y_test.apply(lambda x: x if x in label_encoder.classes_ else None).dropna()
X_test = X_test[y_test.index]
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, padding='post', maxlen=maxlen)
y_test_encoded = label_encoder.transform(y_test)

# 인코딩된 감정 확인
print("Encoded classes: ", label_encoder.classes_)

# CNN+RNN 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=20000, output_dim=128, input_length=maxlen),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 조기 종료와 학습률 감소 콜백
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# 모델 학습
history = model.fit(X_train_pad, y_train_encoded, epochs=20, validation_data=(X_test_pad, y_test_encoded), batch_size=64,
                    callbacks=[early_stopping, reduce_lr])

# 학습 및 검증 정확도 출력
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Validation Accuracy: {val_accuracy:.2f}')

# 예측 함수
def predict_emotion():
    input_text = input_text_box.get("1.0", tk.END).strip()
    if input_text:
        seq = tokenizer.texts_to_sequences([input_text])
        padded = pad_sequences(seq, padding='post', maxlen=maxlen)
        prediction = model.predict(padded)
        predicted_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
        result_label.config(text=f"Predicted Emotion: {predicted_emotion}")

# UI 생성
root = tk.Tk()
root.title("Emotion Analysis AI")
root.geometry("500x400")

# 프레임 생성
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=10, pady=10)

# 입력 텍스트 박스
input_text_box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=40, height=10, font=("Arial", 12))
input_text_box.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# 예측 버튼
predict_button = tk.Button(frame, text="Analyze Emotion", command=predict_emotion, bg="blue", fg="white", font=("Arial", 12, "bold"))
predict_button.grid(row=1, column=0, padx=10, pady=10, sticky="e")

# 결과 레이블
result_label = tk.Label(frame, text="Predicted Emotion: None", font=("Arial", 12, "italic"), fg="green")
result_label.grid(row=1, column=1, padx=10, pady=10, sticky="w")

root.mainloop()
