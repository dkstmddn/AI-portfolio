# 데이터 전처리
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# 엑셀 파일 읽기
data = pd.read_excel('shgmlfo.xlsx')

# 데이터 확인
print(data.head(57))

# 텍스트와 라벨 분리
texts = data['관람평']
labels = data['감정']

# 라벨 인코딩
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

# 한국어 불용어 목록 (예시)
korean_stopwords = [
    '이', '그', '저', '것', '수', '등', '들', '및', '의', '가', '을', '를', 
    '은', '는', '에', '와', '과', '한', '하다', '그리고', '또한', '그러나', '하지만'
]

# 텍스트 벡터화
vectorizer = TfidfVectorizer(stop_words=korean_stopwords, max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 인공지능 모델 학습
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# 모델 평가
y_pred = model.predict(X_test_vectorized)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 모델 저장
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Flask 서버 설정
app = Flask(__name__)
CORS(app)

model_path = os.path.join(os.getcwd(), 'sentiment_model.pkl')
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
label_encoder_path = os.path.join(os.getcwd(), 'label_encoder.pkl')

# 모델 로드
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
label_encoder = joblib.load(label_encoder_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None or 'text' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    text = data['text']
    
    # 텍스트 벡터화
    text_vectorized = vectorizer.transform([text])
    
    # 감정 예측
    prediction = model.predict(text_vectorized)
    sentiment = label_encoder.inverse_transform(prediction)
    
    return jsonify({'sentiment': sentiment[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001)