# Mini project

구현할 모델 :  원하는 특성을 입력하면 그와 비슷한 와인을  추천해주는 챗봇

1. **와인 추천 챗봇 [https://wine-recommender-chatbot.streamlit.app/](https://wine-recommender-chatbot.streamlit.app/)**

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/311427e1-e84a-4cbd-be1d-b5ae2ea83a93.png)

### 데이터셋

[https://www.kaggle.com/datasets/zynicide/wine-reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews)

### 컬럼 특성 분석 및 선택

- country : 와인 생산국
- description : 평가
- designation : 와인을 생산한 양조장 안의 포도밭
- points :  와인을 1-100점 척도로 평가한 WineEnthusiast 점수
- price : 가격
- province : 와인을 생산한 주
- region_1 : 와인을 숙성한 주
- region_2 : 와인을 숙성한 주의 더 특정된 구역
- taster_name : 평가자 이름
- taster_twitter_handle : 평가자 트위터 주소
- title : 와인의 이름
- variety : 포도 품종
- winery : 양조장 이름

**선택한 컬럼**

- country
- description
- price
- title

### 데이터 시각화

### 결측치 제거

```python
# 데이터셋 불러오기
import pandas as pd
df1 = pd.read_csv('/content/drive/MyDrive/엔코어/winemag-data-130k-v2.csv')
df1
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled.png)

```python
# 전체데이터 결측치 확인
df1.isna().sum()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%201.png)

```python
# 사용할 데이터 남기고 컬럼 드랍하기

df1.drop(columns='Unnamed: 0	', inplace = True)
df1.drop(columns='region_1', inplace = True)
df1.drop(columns='region_2', inplace = True)
df1.drop(columns='taster_twitter_handle', inplace = True)
df1.drop(columns='taster_name', inplace = True)
df1.drop(columns='designation', inplace = True)
df1
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%202.png)

country 의 결측치와 province의 결측치가 63으로 같았고 winery와 title의 결측치가 없었기 때문에 winery와 title로 둘의 결측치를 채우기로 결정. 

```python
len(df1[df1['country'].isna()]), len(df1[df1['province'].isna()]), len(df1[df1['winery'].isna()])
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%203.png)

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%204.png)

**위 결측치 중 winery가 ‘**Gotsa Family Wines’인 값들을 추출

```python
df2[df2['winery'] == 'Gotsa Family Wines']
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/f5bcd8e0-f853-46bb-a158-8be085d9a4f1.png)

null값의 이름을 구글에 검색 후 country와 province의 결측치값을 입력

```python
mask = df2['winery'] == 'Gotsa Family Wines'
df2.loc[mask, 'country'] = 'Georgia'
df2.loc[mask, 'province'] = 'Georgia'
df2[df2['winery'] == 'Gotsa Family Wines']
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%205.png)

위와 같은 방법으로 country와 province의 결측치 제거

```python
df2.isna().sum()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%206.png)

### wine type을 결정짓기

각 와인들의 특성을 ‘variety’, ‘title’에서 찾아서 각 와인의 타입 구

```python
# 와인의 종류를 결정하는 함수
def determine_wine_type(row):
    variety = str(row.get("variety", "")).lower()
    title = str(row.get("title", "")).lower()
    # Sparkling Wine 조건 추가
    if any(keyword in variety for keyword in ['spark', 'brut', 'prosecco', 'champagne']) or \
         any(keyword in title for keyword in ["sparkling", "prosecco", "champagne"]):
        return "Sparkling"
    # Red Wine 조건 추가
    elif any(keyword in variety for keyword in ['cabernet sauvignon', 'merlot', 'pinot noir', 'syrah', 'malbec', 'tempranillo', 'sangiovese',
                     'nebbiolo', 'grenache', 'zinfandel', 'barbera', 'petite sirah', 'carmenere', 'petit verdot',
                     'nero d\'avola', 'montepulciano', 'aglianico', 'tannat', 'pinotage', 'dolcetto', 'bonarda',
                     'frappato', 'cesanese', 'negroamaro', 'nerello mascalese', 'sagrantino', 'lagrein', 'primitivo',
                     'mourvèdre', 'carignan', 'touriga nacional', 'tinta roriz', 'garnacha', 'monastrell', 'bobal',
                     'xinomavro', 'agiorgitiko', 'corvina', 'rondinella', 'molinara', 'teroldego', 'refosco', 'schiava',
                     'raboso', 'cesanese d\'affile', 'picpoul', 'gaglioppo', 'nerello cappuccio', 'cannonau', 'pugnitello',
                     'garnacha tintorera', 'graciano', 'tinta miúda', 'bastardo', 'alicante bouschet', 'rufete',
                     'tinta del toro', 'corvina', 'trincadeira', 'prieto picudo', 'tempranillo blend', 'meritage',
                     'red blend', 'rhône-style red blend', 'bordeaux-style red blend', 'cabernet blend',
                     'garnacha blend', 'grenache blend' ]) or \
       any(keyword in title for keyword in ["red", "noir", "merlot", "cabernet", "shiraz"]):
        return "Red"
    # White Wine 조건 추가
    elif any(keyword in variety for keyword in ['chardonnay', 'sauvignon blanc', 'riesling', 'pinot gris', 'pinot grigio', 'gewürztraminer',
                       'viognier', 'chenin blanc', 'albariño', 'grüner veltliner', 'verdejo', 'vermentino',
                       'marsanne', 'roussanne', 'grenache blanc', 'picpoul', 'melon', 'torrontés', 'moscato', 'muscat',
                       'muscat blanc à petits grains', 'fumé blanc', 'müller-thurgau', 'silvaner', 'semillon', 'cortese',
                       'fiano', 'garganega', 'verdicchio', 'grecheetto', 'friulano', 'ribolla gialla', 'kerner', 'sylvaner',
                       'welschriesling', 'scheurebe', 'siegerrebe', 'riesling', 'johannisberg riesling', 'savagnin',
                       'furmint', 'malvasia', 'assyrtiko', 'aidani', 'roditis', 'malagousia', 'robola', 'moschofilero',
                       'white riesling', 'muscat ottonel', 'muskat', 'gelber muskateller', 'sauvignonasse', 'arneis',
                       'erbaluce', 'favorita', 'grechetto', 'pigato', 'timorasso', 'vernaccia', 'verduzzo', 'vermentino nero',
                       'ribolla gialla', 'früburgunder', 'weissburgunder', 'morillon', 'roter veltliner', 'zierfandler',
                       'rotgipfler', 'neuburger', 'blanc du bois', 'chenin blanc-chardonnay', 'chenin blanc-sauvignon blanc',
                       'alsace white blend', 'rhône-style white blend', 'bordeaux-style white blend', 'white blend',
                       'austrian white blend']) or \
         any(keyword in title for keyword in ["white", "gris", "chardonnay", "sauvignon", "pinot blanc"]):
        return "White"
    # Rosé Wine 조건 추가
    elif any(keyword in variety for keyword in ["rosé", "rose", "rosato", "rosado"]) or \
         any(keyword in title for keyword in ["rosé", "rose"]):
        return "Rosé"
    else:
        return "Unknown"
# 'wine_type' 컬럼 생성
df["wine_type"] = df.apply(determine_wine_type, axis=1)
df.head()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%207.png)

```python
df['wine_type'].value_counts()
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%208.png)

와인 type을 예측하지 못한 Unknown을 제거, 동일한 내용의 데이터 제거

```python
df_sorted = df_unique.sort_values(by=['title', 'price'])
unique_title_rows = df_sorted.drop_duplicates(subset=['title'], keep='first')
df_filtered  = unique_title_rows[unique_title_rows['wine_type'] != 'Unknown']

df_filtered  `
```

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%209.png)

```python
# 예측한 값과 실제 값과 일치하는 비율 확인
matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2010.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2011.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2012.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2013.png)

```python
# 원핫 인코딩 결과로 출력된걸 각 번호에 맞는 와인으로 변경
wine_type_map = {0: 'Red', 1: 'Sparkling', 2: 'Rose', 3: 'White'}

# 예측 결과를 와인 타입으로 변환하여 DataFrame에 추가
df['predicted_wine_type'] = df['predicted_wine_type'].map(wine_type_map)

# 결과를 출력합니다.
print(df[['text_combined', 'predicted_wine_type']])
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2014.png)

```python
# 예측한 값과 실제 값과 일치하는 비율 확인

matching_rows = df[df['predicted_wine_type'] == df['wine_type']]
matching_ratio = len(matching_rows) / len(df) * 100

print(f"예측한 와인 타입과 실제 와인 타입이 일치하는 비율: {matching_ratio:.2f}%")
```

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2015.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2016.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2017.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2018.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2019.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2020.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2021.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2022.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2023.png)

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

![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2024.png)

## 와인 추천 챗봇 만들기

---

### **1. 와인 추천 챗봇 [https://wine-recommender-chatbot.streamlit.app/](https://wine-recommender-chatbot.streamlit.app/)**

1. **Streamlit 사용** 

간단한 파이썬 코드를 사용하여 빠르고 쉽게 웹 애플리케이션을 구*축할 수 있는 파*이썬 라이브러리

[Streamlit • A faster way to build and share data apps](https://streamlit.io/)

1. **챗봇 미리보기**

- wine_type 필터링
    
    ![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/95f38d5f-5dfc-4461-8d22-1599b0afe1fd.png)
    

- price 필터링
    
    사용자가 ["price", "cost", "priced", "dollar", "dollars", "$"] 등의 가격을 의미하는 단어를 입력하면 사이드 바에 가격대를 선택할 수 있는 셀렉트박스가 나온다.
    
    ![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2025.png)
    

- description(와인 리뷰)  등 데이터셋 값과 사용자 입력값 유사도 판단
    
    
    TF-IDF를 사용하여 텍스트 벡터화
    
    자주 사용되는 단어가 문서에서 중요하다고 가정하여 문서의 특징을 추출하는 방법으로, 메모리와 계산량이 적고 구현이 쉽다.
    
    코사인 유사도를 사용하여 유사도 판단
    
    텍스트 데이터에서 유사성을 평가하는 데에 효과적이며, 자카드 지수나 유클리드 거리보다 어휘 유사성을 더 잘 고려하여 일반적으로 텍스트 데이터에 대한 유사성 작업에서는 코사인 유사도가 선호 된다.
    
    테스트로 similarity 컬럼을 추가하여 유사도 수치를 확인
    
    ![Untitled](Mini%20project%2046aaa8415690450ea6d0ece0d145ebc2/Untitled%2026.png)
    
1. **배포**
- github repository 생성 [https://github.com/](https://github.com/)
    
    반드시 Public으로 생성
    
- 파이썬 파일과 데이터셋파일 업로드
    
    **github 브라우저를 통해서 직접 올리면 파일당 25MB 제한,  명령줄을 이용하면 최대 100MB 추가 가능,** 100MB 이상이면 Git LFS(Git 대용량 파일 스토리지)를 이용해야 한다.
    
- streamlit 홈페이지에서 회원가입 및 git 계정과 연동 진행 [https://share.streamlit.io/](https://share.streamlit.io/)
- create app을 눌러 배포 진행

1. **정리**

완벽하진 않지만 완성해보는 것에 초점을 맞췄다.

금방 끝날 것 같았던 배포에 거의 하루를 다 써버렸다.
이유는 아무 생각 없이 브라우저를 통해서 직접 37MB 이상인 데이터 셋을 github에 옮기다가 

제한이 걸렸고 명령줄을 통해 할 생각은 못 하고 git LFS, 구글 드라이브 공유 등등 다른 방법을 찾고 또 찾고...
난관에 부딪혔을 때는 제일 먼저 공식 문서를 잘 활용하자

지금은 country가 컬럼의 값과 정확하게 매칭되어야만 필터링 되는데 수정이 필요하고, 

price도 가격을 의미하는 사용자 입력값을 파악해 셀렉트박스 없이 바로 필터링이 될 수 있도록 개선해보고 싶다.

또한 텍스트 임베딩, 유사도 판단 모델을 좀 더 다양하게 적용해 성능을 높여봐야겠다.

---