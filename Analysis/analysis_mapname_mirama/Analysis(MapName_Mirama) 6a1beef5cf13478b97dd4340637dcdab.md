# Analysis(MapName_Mirama)

![그림1. whitezone의 중심](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled.png)

그림1. whitezone의 중심

![그림2. alive_user들의 기하학적 중심점](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%201.png)

그림2. alive_user들의 기하학적 중심점

![그림3. hitezone의 중심과 alive_user들의 중심의 거리](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%202.png)

그림3. hitezone의 중심과 alive_user들의 중심의 거리

```bash
주제 : 배틀그라운드: 생존자 위치와 화이트존 생성의 통계적 상관성 분석
주제선정이유 : 배틀그라운드를 자주 플레이하는 유저로서, 게임의 승패에 큰 영향을 미치는 자기장(세이프티존)의 위치가 무작위로 생성되는지, 
							 아니면 생존자들의 위치를 기반으로 생성되는지에 대해 궁금증이 생겼습니다. 이러한 의문을 해결하기 위해 데이터 분석을 통해 자기장 생성 규칙을 검증하고, 
							 이를 위한 데이터 수집 및 처리 파이프라인을 구축하는 것을 주제로 선정하게 되었습니다.
	
1. 상관분석 수행을 위해 상관계수 사용
 - 피어슨
	- 선형 상관계수 측정
	- 데이터가 정규 분포를 따를때 사용
	- 두 변수 간의 선형관계 측정
 - 스피어만
	- 데이터 분포에 대한 가정이 필요없음
	- 데이터가 정규분포를 따르지 않아도 유용함
2. 상관 계수 선택을 위한 데이터의 정규성 검정 : 데이터의 정규분포 여부확인
 - 시각화 : 히스토그램, QQ플롯, BOX 플롯
 - 통계적 방법 : 샤피로-윌크 테스트, 콜모그라브-스미노프 테스트
 - H0(귀무가설): 살아있는 유저들의 위치와 자기장 중심의 위치는 무관하다. 즉, 자기장은 무작위로 잡힌다.
 - H1(대립가설): 살아있는 유저들의 위치가 자기장 중심의 위치에 영향을 미친다. 즉, 자기장은 살아있는 유저들의 위치를 고려하여 잡힌다.
	- P-value : 값이 작을수록 귀무가설 기각 근거 강해짐
	- alpha(유의수준) : 귀무가설을 기각하는 기준이 되는 값 0.05 또는 0.01
```

```bash
import requests
import matplotlib.pyplot as plt
from scipy.stats import shapiro, spearmanr, probplot
import numpy as np

# GET 요청을 보낼 기본 URL
base_url = 'http://192.168.0.79:5000'
endpoints = "/whitezoneAnalysis/phases"
header = {
    "Accept": "application/json"
}

# GET 요청 보내기
get_response = requests.get(url=base_url + endpoints, headers=header)

# 응답 상태 코드 확인
print('Status Code:', get_response.status_code)

# 응답 데이터 출력
if get_response.status_code == 200:
    data = get_response.json()
else:
    print('Failed to retrieve data')
    data = {}

# 거리 리스트 초기화
distance_list = []
phase_coordinates = []

for i in range(len(data)):
    phase_list = []
    coordinates_list = []
    phase_key = f"Phase{i+1}"
    if phase_key in data:
        for j in range(len(data[phase_key])):
            x1 = data[phase_key][j]["user_geometry_center_x"]
            x2 = data[phase_key][j]["white_zone_center_x"]
            y1 = data[phase_key][j]["user_geometry_center_y"]
            y2 = data[phase_key][j]["white_zone_center_y"]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            phase_list.append(distance)
            coordinates_list.append((x1, y1, x2, y2))
        distance_list.append(phase_list)  # 반복문 밖으로 이동
        phase_coordinates.append(coordinates_list)

# 이상치 제거 함수 (상한치만 고려)
def remove_upper_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3.5 * IQR
    return [x for x in data if x <= upper_bound]

# 초기 x축 끝값 설정 및 페이즈별 줄어드는 비율
x_end_values = [400000, 219970.803617571057, 131740.25348840904, 78900.030827725155, 51230.391445946857, 33270.6639480441026, 21600.2976675614, 15100.0693623677593]

# 각 페이즈별로 샤피로-윌크 검정 수행 및 히스토그램 시각화
num_phases = len(distance_list)
fig, axes = plt.subplots(2, num_phases, figsize=(25, 10))

# 상관분석 결과 저장
# 가설 설명
print("""
- H0(귀무가설): 데이터는 정규분포를 따른다.
- H1(대립가설): 데이터는 정규분포를 따르지 않는다.
""")
correlation_results = []

for i in range(num_phases):
    specific_phase = remove_upper_outliers(distance_list[i])  # 상한치만 고려하여 이상치 제거
    
    # 샤피로-윌크 검정
    stat, p_value = shapiro(specific_phase)
    print(f'Phase {i+1} - Shapiro-Wilk Test: Statistics={stat}, p-value={p_value}')
    if p_value > 0.05:
        print(f'Phase {i+1} 데이터는 정규 분포를 따릅니다 (귀무가설 기각 실패)')
    else:
        print(f'Phase {i+1} 데이터는 정규 분포를 따르지 않습니다 (귀무가설 기각)')

    # 히스토그램 그리기
    x_end = x_end_values[i]
    bins = np.linspace(0, x_end, 35)  # 35개의 bin으로 나누기
    
    counts, bins, _ = axes[0, i].hist(specific_phase, bins=bins, edgecolor='black', alpha=0.5)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    axes[0, i].plot(bin_centers, counts, linestyle='-', marker='o', color='blue')
    axes[0, i].set_title(f'Histogram of Phase {i+1}')
    axes[0, i].set_xlabel('Distance')
    axes[0, i].set_ylabel('Frequency')
    axes[0, i].set_xlim(0, x_end)
    
    # x축 라벨링 변경
    locator = plt.FixedLocator(bin_centers)
    labels = [str(idx + 1) for idx in range(len(bin_centers))]
    formatter = plt.FixedFormatter(labels)
    axes[0, i].xaxis.set_major_locator(locator)
    axes[0, i].xaxis.set_major_formatter(formatter)
    
    # bin_width 출력
    print(f"Phase {i+1} - Bin Width: {x_end / 35}")

    # QQ 플롯 생성
    probplot(specific_phase, dist="norm", plot=axes[1, i])
    axes[1, i].set_title(f'QQ Plot of Phase {i+1}')
    axes[1, i].set_xlabel('Theoretical Quantiles')
    axes[1, i].set_ylabel('Sample Quantiles')
    
    
    # 스피어만 상관분석
    corr, p_value = spearmanr(bin_centers, counts)
    correlation_results.append((i + 1, corr, p_value))

plt.tight_layout()
plt.show()

# 가설 설명
print("""
- H0(귀무가설): 살아있는 유저들의 위치와 자기장 중심의 위치는 무관하다. 즉, 자기장은 무작위로 잡힌다.
- H1(대립가설): 살아있는 유저들의 위치가 자기장 중심의 위치에 영향을 미친다. 즉, 자기장은 살아있는 유저들의 위치를 고려하여 잡힌다.
""")
# 스피어만 상관분석 결과 출력
for result in correlation_results:
    phase, corr, p_value = result
    print(f'Phase {phase} - 스피어만 상관분석: 상관계수={corr}, p-value={p_value}')

```

- 그림3 의 히스토그램

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%203.png)

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%204.png)

## 정규분포 확인

1. 샤피로-윌크 검정

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%205.png)

1. QQ-plot

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%206.png)

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%207.png)

[각 Phase별 분석]

1. **Phase 1**:
    - QQ 플롯에서 대부분의 데이터 점들이 대각선에 가까이 분포하고 있으나, 꼬리 부분에서 대각선에서 벗어나 있는 점들이 있습니다.
    - 이는 Phase 1의 데이터가 전체적으로 정규분포에 근접하지만, 극단값(outlier)에서 정규분포를 따르지 않는다는 것을 의미합니다.
2. **Phase 2**:
    - QQ 플롯에서 중간 구간의 데이터 점들은 대각선에 가깝게 분포하고 있으나, 꼬리 부분에서 대각선에서 벗어나 있습니다.
    - Phase 2의 데이터는 중심부에서는 정규분포를 따르지만, 극단값에서는 정규분포를 따르지 않습니다.
3. **Phase 3**:
    - 중간 구간의 데이터 점들이 대각선에 근접하지만, 극단값에서 벗어나 있는 경향이 있습니다.
    - Phase 3의 데이터는 중심부에서는 정규분포에 가깝지만, 꼬리 부분에서는 정규분포를 따르지 않습니다.
4. **Phase 4**:
    - 중간 구간의 데이터 점들이 대각선에 가깝게 분포하지만, 끝부분에서 벗어나는 경향이 있습니다.
    - Phase 4의 데이터는 중심부에서는 정규분포를 따르지만, 극단값에서는 정규분포를 따르지 않습니다.
5. **Phase 5**:
    - 중간 구간의 데이터 점들이 대각선에서 벗어나고 있으며, 꼬리 부분에서 대각선에서 크게 벗어나 있습니다.
    - Phase 5의 데이터는 정규분포를 따르지 않는다는 것을 나타냅니다.
6. **Phase 6**:
    - QQ 플롯에서 데이터 점들이 대각선에서 크게 벗어나고 있습니다.
    - 이는 Phase 6의 데이터가 정규분포를 따르지 않는다는 것을 의미합니다.
7. **Phase 7**:
    - 데이터 점들이 대각선에서 크게 벗어나고 있습니다.
    - Phase 7의 데이터는 정규분포를 따르지 않는 것으로 보입니다.
8. **Phase 8**:
    - 데이터 점들이 대각선에서 벗어나 있으며, 특히 극단값에서 크게 벗어나고 있습니다.
    - Phase 8의 데이터는 정규분포를 따르지 않는 것으로 보입니다.

### 결론:

- 모든 Phase에서 QQ 플롯을 보면, 중간 구간에서는 대체로 대각선에 근접하지만 꼬리 부분에서 크게 벗어나는 경향이 있습니다.
- 이는 모든 Phase의 데이터가 정규분포를 따르지 않음을 시사합니다.
- 이는 샤피로-윌크 검정 결과와도 일치합니다.
- 따라서 모든 Phase에서 데이터는 정규분포를 따르지 않는 것으로 분석됩니다.

상관관계 분석을 위해 정규분포를 알아본 결과 정규분포를 따르지 않기에 상관분석은 스피어만 상관분석으로 결정하였음. 

## 상관관계 검증

![Untitled](Analysis(MapName_Mirama)%206a1beef5cf13478b97dd4340637dcdab/Untitled%208.png)

### 각 페이즈별 분석 결과:

- **Phase 1:**
    - 상관계수: -0.640572376133425
    - p-value: 4.490554635969112e-05
    - 해석: 유의미한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 유의미한 음의 상관관계가 존재합니다.
- **Phase 2:**
    - 상관계수: -0.6434290218175496
    - p-value: 4.0480149725718794e-05
    - 해석: 유의미한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 유의미한 음의 상관관계가 존재합니다.
- **Phase 3:**
    - 상관계수: -0.7489504124696674
    - p-value: 3.488211673978302e-07
    - 해석: 강한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 강한 음의 상관관계가 존재합니다.
- **Phase 4:**
    - 상관계수: -0.7350075512021718
    - p-value: 7.412121803432713e-07
    - 해석: 강한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 강한 음의 상관관계가 존재합니다.
- **Phase 5:**
    - 상관계수: -0.7321263032134597
    - p-value: 8.611463651503739e-07
    - 해석: 강한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 강한 음의 상관관계가 존재합니다.
- **Phase 6:**
    - 상관계수: -0.6048640678104168
    - p-value: 0.0001511163942786868
    - 해석: 유의미한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 유의미한 음의 상관관계가 존재합니다.
- **Phase 7:**
    - 상관계수: -0.5892087551875876
    - p-value: 0.00024606444617736344
    - 해석: 유의미한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 유의미한 음의 상관관계가 존재합니다.
- **Phase 8:**
    - 상관계수: -0.7006037456435822
    - p-value: 3.9598520616432805e-06
    - 해석: 강한 음의 상관관계가 있습니다. 즉, 히스토그램의 x값(거리)과 y값(빈도수) 간에 강한 음의 상관관계가 존재합니다.

### 결론:

- 각 페이즈의 상관계수가 모두 음수이고, p-value가 0.05보다 작아 귀무가설(H0)을 기각할 수 있습니다. 이는 살아있는 유저들의 위치와 자기장 중심의 위치 간에 유의미한 음의 상관관계가 있음을 의미합니다.
- 상관계수가 음수라는 것은 거리가 멀어질수록 빈도수가 줄어드는 경향이 있음을 나타냅니다.
- 따라서, 자기장의 위치는 살아있는 유저들의 위치에 영향을 받으며, 유저들이 몰린 곳에서 멀어질수록 빈도수가 낮아짐을 알 수 있습니다.