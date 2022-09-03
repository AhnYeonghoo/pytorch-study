import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


print(sklearn.__version__)
dataset = pd.read_csv("C:/Users/dksdu/OneDrive/바탕 화면/project/PythonMLWorkspace/ScitkitLearn/LinearRegressionData.csv")
print(dataset.head())
print()
print()
X = dataset.iloc[:, :-1].values # 처음부터 마지막 칼럼 직전까지의 데이터(독립 변수)
y = dataset.iloc[:, -1].values
print(X)
print(y)

from sklearn.linear_model import LinearRegression

reg = LinearRegression() # 선형회귀 인스턴스 생성
reg.fit(X, y) # 학습 (모델 생성)
y_pred = reg.predict(X) # X에 대한 예측값
print(y_pred)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='green')
plt.title("Score by hours")
plt.xlabel('hours') # X축 이름 
plt.ylabel('score') # y축 이름
plt.show()


print(f"8시간 공부했을 때 예상 점수: {reg.predict([[8], [9], [1]])}")
print(reg.coef_) # 기울기
print(reg.intercept_) # y절편
# y = mx + b -> y = 10.444x, y = -0.2185

