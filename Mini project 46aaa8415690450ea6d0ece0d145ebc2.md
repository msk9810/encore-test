# Mini project

1. 구현할 모델 :  원하는 특성을 입력하면 그와 비슷한 와인을  추천해주는 챗봇

### **1.**  Tf-idf와 **LogisticRegression을 이용해 학습한 모델로 와인을 추측한 결과**

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')

# 결측치 제거
df.dropna(inplace=True)

# 텍스트 데이터 결합
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['text_combined'])

# 분류 모델 학습
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, df['wine_type'])

# 예측
df['predicted_wine_type'] = logreg.predict(X)

# 추가 조건에 따라 예측값 수정
df.loc[df['text_combined'].str.contains('spark', case=False), 'predicted_wine_type'] = 'Sparkling'
# df.loc[df['text_combined'].str.contains('rose', case=False), 'predicted_wine_type'] = 'Rose'

# 결과 확인
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled.png)

```python
# 예측한 값과 실제 값과 일치하는 비율 확인
matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%201.png)

```python
# predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%202.png)

### **2.  Keras의 딥러닝을 이용해 학습한 모델로 와인을 추측한 결과**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# 토크나이저 설정
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

# 텍스트를 시퀀스로 변환
X_seq = tokenizer.texts_to_sequences(X)

# 시퀀스 길이를 맞춰주기 위해 패딩 추가
X_pad = pad_sequences(X_seq, maxlen=100)

# 타겟 데이터 원-핫 인코딩
y_encoded = pd.get_dummies(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_pad, y_encoded, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(Embedding(5000, 128, input_length=100))
model.add(LSTM(128))
model.add(Dense(4, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%203.png)

```python
# 학습시킨 모델로 전체 데이터를 예측해본 결과
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/my_wine_data.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 토크나이저 설정
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text_combined'])

# 텍스트를 시퀀스로 변환
X_seq = tokenizer.texts_to_sequences(df['text_combined'])

# 시퀀스 길이를 맞춰주기 위해 패딩 추가
X_pad = pad_sequences(X_seq, maxlen=100)

# 모델을 사용하여 예측 수행
predictions = model.predict(X_pad)

# 예측 결과를 분석하여 필요한 처리를 수행합니다.
# 여기서는 각 예측 값의 인덱스를 사용하여 해당하는 와인 타입을 찾을 수 있습니다.

# 예측 결과를 DataFrame에 추가
df['predicted_wine_type'] = predictions.argmax(axis=1)

# 결과를 저장하거나 출력합니다.
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%204.png)

```python
# 원핫 인코딩 결과로 출력된걸 각 번호에 맞는 와인으로 변경
wine_type_map = {0: 'Red', 1: 'Sparkling', 2: 'Rose', 3: 'White'}

# 예측 결과를 와인 타입으로 변환하여 DataFrame에 추가
df['predicted_wine_type'] = df['predicted_wine_type'].map(wine_type_map)

# 결과를 출력합니다.
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%205.png)

```python
# 예측한 값과 실제 값과 일치하는 비율 확인

matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%206.png)

```python
# predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%207.png)

### **3.**  Tf-idf와 SVC를 이용해 학습한 모델로 예측한 결과

```python
# SVC를 이용해서 분할 해서 test돌린 결과값 : (Accuracy: 0.9889141425854632)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v1.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# 테스트 세트에서 예측
y_pred = svm_model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%208.png)

```python
# 1차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%209.png)

```python
# 1차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9943318335729857)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v1.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2010.png)

```python
# 1차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2011.png)

```python
# 2차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9987012384618953)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/drive/MyDrive/엔코어/wine_data_with_predictions_v3.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2012.png)

```python
# 2차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2013.png)

```python
# 3차로 학습시킨 결과값을 이용해서 다시 전체를돌린 결과값 : (Accuracy: 0.9993227886265597)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 데이터셋 로드
df = pd.read_csv('/content/wine_data_with_predictions_v4.csv')
df['text_combined'] = df['description'] + ' ' + df['title'] + ' ' + df['variety']

# 입력과 타겟 데이터 설정
X = df['text_combined']
y = df['wine_type']

# TF-IDF 벡터화
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# SVM 모델 정의 및 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_tfidf, y)

# 전체 데이터에 대한 예측
y_pred = svm_model.predict(X_tfidf)

# 정확도 계산
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2014.png)

```python
# 3차로 나온 predicted_wine이랑 wine_type이랑 비교 시각화

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 혼동 행렬 생성
cm = confusion_matrix(df['wine_type'], df['predicted_wine_type'])

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(df['wine_type'].unique()),
            yticklabels=sorted(df['wine_type'].unique()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2015.png)

와인 추천 챗봇 만들기

Streamlit 

실시간 대화형 Web 애플리케이션을 쉽게 만들 수 있는 패키지

[Streamlit • A faster way to build and share data apps](https://streamlit.io/)