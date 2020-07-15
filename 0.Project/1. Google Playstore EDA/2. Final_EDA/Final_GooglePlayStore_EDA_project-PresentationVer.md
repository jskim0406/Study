# Google Play Store data EDA 분석

## 데이터 소개

# Google Play Store Apps

저희가 사용한 data는 Google Play Store에서 수집된 데이터로 Kaggle에 올라와 있는 데이터를 사용

https://www.kaggle.com/lava18/google-play-store-apps


이 데이터는:
- 총 13개의 특징 칼럼
- 9660개의 unique한 값
- 총 10842개의 데이터
- csv 파일 
- 1개를 제외한 12개의 특징 칼럼들은 모두 object 타입

# 데이터 전처리 

1. 결측치 확인
    Rating, Version data 결측치 존재
    - Rating : 1,473개 데이터 -> NaN값 데이터 존재
        => 모두 0으로 처리 (사용자의 평가 유보 대상 서비스 판단, Install정보 기반) 
        * Installs수( Rating 0 mean / 전체 mean / 전체 median) : (4,095, 1,546만, 10만)

    - Version : Current Ver 8개 / Android Ver 2개 
        => 모두 0으로 처리 


2. 이상치 제거
    2개 행(데이터) 이상치 제거
    - 총 rows 수 : 10841개 -> 10839개


3. 컬럼별 데이터타입 변경
    Size, Reviews, Installs, Price, Rating : 숫자로 변경


4. 컬럼 추가
    - Log 값 적용한 컬럼 생성 : Installs_log, Reviews_log
    - 추가 이유 : 데이터의 증감 추세 유지 하 데이터 수치 너비를 좁혀 분석 용이성 제고


# 컬럼 별 데이터 분석


### 1. Category


### Top 10 App Categories


```python
plt.figure(figsize=(15,6))
sns.barplot(x=category.index[:10], y ='Count',data = category[:10],palette='hls')
plt.title('Top 10 App categories')
plt.xticks(rotation=90)
plt.show()
```

![output_9_0](https://user-images.githubusercontent.com/14319885/85916789-f8f69d80-b88e-11ea-98e4-1b6fc011125c.png)


### Finding

    1) 가장 많은 카테고리 : Family(18%), Game(11%)
    2) 가장 적은 카테고리 : Beauty(1%), Cosmic(1%)


### 2. Rating



```python
#각 Rating별 갯수
plt.subplots(figsize=(10,10))
plt.xticks(rotation=90)
ax = sns.countplot(x="Rating", data=df, palette="Set3")
```

![output_12_0](https://user-images.githubusercontent.com/14319885/85916804-0d3a9a80-b88f-11ea-8cec-5370d3fc5a36.png)


### Finding

    1) 4점 이상의 앱 비중 : 67.97%

### 4.Reviews


```python
#histogram

plt.figure(figsize=(10,5))
sns.distplot(df['Reviews_log'],color='g')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12dea2e10>



![output_15_1](https://user-images.githubusercontent.com/14319885/85916810-1a578980-b88f-11ea-8192-c229f9da9b2e.png)


### Findings

    1) Review 갯수는 약 1억개까지 분포
    2) Review 갯수 상위 앱은 Facebook (약 1억 5천개)

### 4. Installs


```python
print(df['Installs_log'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Installs_log'], color='g', bins=10, hist_kws={'alpha': 0.4});
```

    count    10839.000000
    mean         4.809655
    std          2.113071
    min          0.000000
    25%          3.000000
    50%          5.000000
    75%          6.698970
    max          9.000000
    Name: Installs_log, dtype: float64



![output_18_1](https://user-images.githubusercontent.com/14319885/85916826-365b2b00-b88f-11ea-9125-ad0cb0205317.png)


### Finding
```
    1) 설치횟수가 1백만 이상인 앱의 비중 : 14.57%, 1천만 이상인 앱의 비중 : 11.55%
    2) 평균 설치 횟수 : 1천 5백만 회
    3) 최대 설치 횟수 : 10억 회
    4) 최소 설치 횟수 : 0회
```

### 6. Price : 무료 앱 제외 후 분석 (무료 앱 갯수 : 790개)


```python
plt.figure(figsize=(8,6))
plt.title('Distribution of Paid App Prices')
sns.distplot(paid_apps['Price'],bins=50)
plt.show()
```

![output_21_0](https://user-images.githubusercontent.com/14319885/85916830-41ae5680-b88f-11ea-81af-3342e38c9930.png)

```python
paid_apps[paid_apps['Price'] >= 350]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
      <th>Reviews_log</th>
      <th>Installs_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4197</th>
      <td>most expensive app (H)</td>
      <td>FAMILY</td>
      <td>4.3</td>
      <td>6</td>
      <td>1.5M</td>
      <td>100</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>July 16, 2018</td>
      <td>1.0</td>
      <td>7.0 and up</td>
      <td>0.778151</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>4362</th>
      <td>💎 I'm rich</td>
      <td>LIFESTYLE</td>
      <td>3.8</td>
      <td>718</td>
      <td>26M</td>
      <td>10000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>March 11, 2018</td>
      <td>1.0.0</td>
      <td>4.4 and up</td>
      <td>2.856124</td>
      <td>4.00000</td>
    </tr>
    <tr>
      <th>4367</th>
      <td>I'm Rich - Trump Edition</td>
      <td>LIFESTYLE</td>
      <td>3.6</td>
      <td>275</td>
      <td>7.3M</td>
      <td>10000</td>
      <td>Paid</td>
      <td>400.00</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>May 3, 2018</td>
      <td>1.0.1</td>
      <td>4.1 and up</td>
      <td>2.439333</td>
      <td>4.00000</td>
    </tr>
    <tr>
      <th>5351</th>
      <td>I am rich</td>
      <td>LIFESTYLE</td>
      <td>3.8</td>
      <td>3547</td>
      <td>1.8M</td>
      <td>100000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>January 12, 2018</td>
      <td>2.0</td>
      <td>4.0.3 and up</td>
      <td>3.549861</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>5354</th>
      <td>I am Rich Plus</td>
      <td>FAMILY</td>
      <td>4.0</td>
      <td>856</td>
      <td>8.7M</td>
      <td>10000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>May 19, 2018</td>
      <td>3.0</td>
      <td>4.4 and up</td>
      <td>2.932474</td>
      <td>4.00000</td>
    </tr>
    <tr>
      <th>5356</th>
      <td>I Am Rich Premium</td>
      <td>FINANCE</td>
      <td>4.1</td>
      <td>1867</td>
      <td>4.7M</td>
      <td>50000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>November 12, 2017</td>
      <td>1.6</td>
      <td>4.0 and up</td>
      <td>3.271144</td>
      <td>4.69897</td>
    </tr>
    <tr>
      <th>5357</th>
      <td>I am extremely Rich</td>
      <td>LIFESTYLE</td>
      <td>2.9</td>
      <td>41</td>
      <td>2.9M</td>
      <td>1000</td>
      <td>Paid</td>
      <td>379.99</td>
      <td>Everyone</td>
      <td>Lifestyle</td>
      <td>July 1, 2018</td>
      <td>1.0</td>
      <td>4.0 and up</td>
      <td>1.612784</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>5358</th>
      <td>I am Rich!</td>
      <td>FINANCE</td>
      <td>3.8</td>
      <td>93</td>
      <td>22M</td>
      <td>1000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>December 11, 2017</td>
      <td>1.0</td>
      <td>4.1 and up</td>
      <td>1.968483</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>5359</th>
      <td>I am rich(premium)</td>
      <td>FINANCE</td>
      <td>3.5</td>
      <td>472</td>
      <td>965k</td>
      <td>5000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>May 1, 2017</td>
      <td>3.4</td>
      <td>4.4 and up</td>
      <td>2.673942</td>
      <td>3.69897</td>
    </tr>
    <tr>
      <th>5362</th>
      <td>I Am Rich Pro</td>
      <td>FAMILY</td>
      <td>4.4</td>
      <td>201</td>
      <td>2.7M</td>
      <td>5000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>May 30, 2017</td>
      <td>1.54</td>
      <td>1.6 and up</td>
      <td>2.303196</td>
      <td>3.69897</td>
    </tr>
    <tr>
      <th>5364</th>
      <td>I am rich (Most expensive app)</td>
      <td>FINANCE</td>
      <td>4.1</td>
      <td>129</td>
      <td>2.7M</td>
      <td>1000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Teen</td>
      <td>Finance</td>
      <td>December 6, 2017</td>
      <td>2</td>
      <td>4.0.3 and up</td>
      <td>2.110590</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>5366</th>
      <td>I Am Rich</td>
      <td>FAMILY</td>
      <td>3.6</td>
      <td>217</td>
      <td>4.9M</td>
      <td>10000</td>
      <td>Paid</td>
      <td>389.99</td>
      <td>Everyone</td>
      <td>Entertainment</td>
      <td>June 22, 2018</td>
      <td>1.5</td>
      <td>4.2 and up</td>
      <td>2.336460</td>
      <td>4.00000</td>
    </tr>
    <tr>
      <th>5369</th>
      <td>I am Rich</td>
      <td>FINANCE</td>
      <td>4.3</td>
      <td>180</td>
      <td>3.8M</td>
      <td>5000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>March 22, 2018</td>
      <td>1.0</td>
      <td>4.2 and up</td>
      <td>2.255273</td>
      <td>3.69897</td>
    </tr>
    <tr>
      <th>5373</th>
      <td>I AM RICH PRO PLUS</td>
      <td>FINANCE</td>
      <td>4.0</td>
      <td>36</td>
      <td>41M</td>
      <td>1000</td>
      <td>Paid</td>
      <td>399.99</td>
      <td>Everyone</td>
      <td>Finance</td>
      <td>June 25, 2018</td>
      <td>1.0.2</td>
      <td>4.1 and up</td>
      <td>1.556303</td>
      <td>3.00000</td>
    </tr>
  </tbody>
</table>
</div>



### Finding
```
    1) 10$ 이하 앱 비중 : 89%
    2) 350$ 이상 앱 : 16개의 앱이 350$ 이상의 Price. 8건 이상이 10,000건 이상의 다운로드 횟수 기록
```

# 컬럼 별 상관관계 분석

    1) Reviews_log - Installs_log
    2) Reviews_log - Rating


```python
df.corr()['Reviews_log']
```




    Rating          0.595889
    Reviews         0.289233
    Installs        0.314721
    Price          -0.028740
    Reviews_log     1.000000
    Installs_log    0.956194
    Name: Reviews_log, dtype: float64



## 1. Reviews_log와 Installs_log 분석

    1) log값 취한 데이터를 기준으로 상관성 분석
    2) 목표하는 Installs 컬럼을 포함한 전체 특징들과 분석




```python
# joint scatter plot

sns.jointplot(x="Installs_log", y="Reviews_log", data=df, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x12d9fd610>



![output_27_1](https://user-images.githubusercontent.com/14319885/85916836-4ffc7280-b88f-11ea-8ccd-9c3be291d58d.png)

```python
df.corr()['Reviews_log']
```




    Rating          0.499426
    Reviews         0.289243
    Installs        0.314730
    Price          -0.028723
    Installs_log    0.956188
    Reviews_log     1.000000
    Name: Reviews_log, dtype: float64



### Finding

```
Reviews수의 로그를 취한 데이터와 상관관계를 보인 데이터
    : Installs_log, Rating

```

## 2. Rating, Reviews_log 선형 분석

    1) log값 취한 데이터를 기준으로 상관성 분석
    2) Reviews와의 상관관계를 보이는 feature에 대한 추가 탐색


```python
j = sns.jointplot(x="Reviews_log", y="Rating", data=df, kind='reg')
j.annotate(stats.pearsonr)
plt.show()
```

![output_31_0](https://user-images.githubusercontent.com/14319885/85916840-5b4f9e00-b88f-11ea-8b19-1ead187c14de.png)


# Reviews 데이터 분석

    1) Reviews가 높은 Install과 Rating의 유의미한 상관관계를 보이는 feature임을 확인
    2) 이에 따라, 별도의 Review의 User_data 활용 분석
    3) 상위 Install 앱의 Reivew 데이터 분석 실시

### Installs_log, Reviews_log의 sorting 결과, 상위 10개 데이터 분석


```python
df1 = df1.sort_values(by=['Installs_log', 'Reviews_log'], ascending=False)
df1.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Category</th>
      <th>Rating</th>
      <th>reviews</th>
      <th>Installs</th>
      <th>Reviews_log</th>
      <th>Installs_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2544</th>
      <td>SOCIAL</td>
      <td>4.1</td>
      <td>78158306.0</td>
      <td>1.000000e+09</td>
      <td>7.892975</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3943</th>
      <td>SOCIAL</td>
      <td>4.1</td>
      <td>78128208.0</td>
      <td>1.000000e+09</td>
      <td>7.892808</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>336</th>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>69119316.0</td>
      <td>1.000000e+09</td>
      <td>7.839599</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>381</th>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>69119316.0</td>
      <td>1.000000e+09</td>
      <td>7.839599</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3904</th>
      <td>COMMUNICATION</td>
      <td>4.4</td>
      <td>69109672.0</td>
      <td>1.000000e+09</td>
      <td>7.839539</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2604</th>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>66577446.0</td>
      <td>1.000000e+09</td>
      <td>7.823327</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2545</th>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>66577313.0</td>
      <td>1.000000e+09</td>
      <td>7.823326</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>2611</th>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>66577313.0</td>
      <td>1.000000e+09</td>
      <td>7.823326</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3909</th>
      <td>SOCIAL</td>
      <td>4.5</td>
      <td>66509917.0</td>
      <td>1.000000e+09</td>
      <td>7.822886</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>382</th>
      <td>COMMUNICATION</td>
      <td>4.0</td>
      <td>56646578.0</td>
      <td>1.000000e+09</td>
      <td>7.753174</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



### Installs, Reviews 상위 10개 App


```python
df_or.iloc[[2544, 3943, 336, 381, 3904, 2604, 2545, 2611, 3909, 382]].App
```




    2544                                    Facebook
    3943                                    Facebook
    336                           WhatsApp Messenger
    381                           WhatsApp Messenger
    3904                          WhatsApp Messenger
    2604                                   Instagram
    2545                                   Instagram
    2611                                   Instagram
    3909                                   Instagram
    382     Messenger – Text and Video Chat for Free
    Name: App, dtype: object



### Install, Review 최다횟수 보유 app인 Facebook의 Review Data 분석

    - Sentiment 위주 분석 진행


```python
df_facebook = df_r.loc[df_r["App"] == 'Facebook']
df_facebook_f = df_facebook.sort_values(by='Sentiment_Polarity', ascending=False)
df_facebook_f.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Translated_Review</th>
      <th>Sentiment</th>
      <th>Sentiment_Polarity</th>
      <th>Sentiment_Subjectivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42887</th>
      <td>Facebook</td>
      <td>The best free</td>
      <td>Positive</td>
      <td>0.700000</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>42904</th>
      <td>Facebook</td>
      <td>Good</td>
      <td>Positive</td>
      <td>0.700000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>42891</th>
      <td>Facebook</td>
      <td>Beautiful and sweet. *</td>
      <td>Positive</td>
      <td>0.600000</td>
      <td>0.825000</td>
    </tr>
    <tr>
      <th>42871</th>
      <td>Facebook</td>
      <td>Do understand hindi? If understand problem mem...</td>
      <td>Positive</td>
      <td>0.500000</td>
      <td>0.625000</td>
    </tr>
    <tr>
      <th>42848</th>
      <td>Facebook</td>
      <td>I updated Facebook still way find marketplace....</td>
      <td>Positive</td>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>42971</th>
      <td>Facebook</td>
      <td>Some error keeps popping everytime I link sayi...</td>
      <td>Positive</td>
      <td>0.472222</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>42940</th>
      <td>Facebook</td>
      <td>Ok I'm kinda fed Facebook right now.... I clue...</td>
      <td>Positive</td>
      <td>0.428571</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>42961</th>
      <td>Facebook</td>
      <td>Would great change constantly. The videos inco...</td>
      <td>Positive</td>
      <td>0.383333</td>
      <td>0.544444</td>
    </tr>
    <tr>
      <th>42967</th>
      <td>Facebook</td>
      <td>Hi Facebook, far awesome. Just hope Facebook p...</td>
      <td>Positive</td>
      <td>0.366667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>42844</th>
      <td>Facebook</td>
      <td>Like wont let post pics groups select pic gall...</td>
      <td>Positive</td>
      <td>0.350000</td>
      <td>0.450000</td>
    </tr>
  </tbody>
</table>
</div>



### Facebook의 Review의 Sentiment 분포도


```python
df_facebook["Sentiment"].value_counts().plot.pie(label='Sentiment', autopct='%1.0f%%', figsize=(2, 2))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21b2969b908>



![output_40_1](https://user-images.githubusercontent.com/14319885/85916851-73272200-b88f-11ea-878d-0275e6b3092b.png)


### Whats_app, Instagram 대한 Review들은 data가 존재하지 않아 분석할 수 없음
    - 이유 : 데이터 부재(App 이름의 'I' 이후 데이터 부재)

# Conclusion

    1) 높은 Installs과 Rating을 위해선 많은 Review 갯수가 중요하다는 관계 발견
    2) Price : Price의 중요도는 상대적으로 낮은 것으로 확인됨 (상관계수 절대값 0.1 이상 feature 부재)
    3) Category : 특정 Category에 치중된 상품 구성 (예상과 달리 Social의 비중이 적음)
    4) Rating : 90% 가까이 4점 rating. 객관적인 평가가 이루어진다는 점에는 의문

