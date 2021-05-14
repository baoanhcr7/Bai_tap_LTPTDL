#Bước 0
'''
Hãy xây dựng mô hình hồi quy tuyến tính cho biến H5 = f(X)
trong đó X là một biến định lượng trong bảng dữ liệu tuyển sinh đại học 
giả sử ở đây X là ?
'''
#Bước 1 import thư viện
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import PredictionError


#Bước 2 Tải dữ liện lên chương trình

df = pd.read_csv('dulieuxettuyendaihoc.csv',header=0,delimiter=',')
df['DT'].fillna(0,inplace=True)
#====> Bước 3: Xử lý missing, noise and error
print(df['H5'].isna().sum())

#Bước 4 Phân tích định lượng biến H5 
print(df['H5'].describe(include = 'all'))
fig = plt.figure(figsize=(10,10))
# Histogram
ax = fig.add_subplot(2,1,1)
df[['H5']].hist(bins=10, ax = ax,color='g',edgecolor='white')

# Box chart
ax = fig.add_subplot(2,1,2)
sns.boxplot(x=df['H5'],color='g')
plt.show()

#Bước 5 Phân tích tương quan biến H5
cov_df = df.cov()
print(cov_df['H5'].sort_values(ascending=False).head())

corr_cf = df.corr()
print(corr_cf['H5'].sort_values(ascending=False).head())
# --> Tìm ra cột định lượng có tương quan cao nhất so với DH1 --> sinh viên tự thử nghiệm
# --> Nếu max(corr_cf) >= 0.5 --> tiến hành phân tích tương quan bước 7
# --> Nếu max(corr_cf) < 0.5 --> Tiến hành gom cụm bước 6 
# --> hy vọng tìm ra cụm có tương quan cao --> Phân tích trên các cụm

'''
H6 có độ tương quan với H5 cao nhất
0.752959 > 0.5 nên không phân cụm
'''
df.plot.scatter(x='H6',y='H5',c='DarkBlue')
#Bước 7: Phân tích hồi quy Linear Regression
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

y_train = np.array(train_df.H5)
X_train = np.array(train_df.H6)
X_train = X_train.reshape(X_train.shape[0], 1)

y_test = np.array(test_df.H5)
X_test = np.array(test_df.H6)
X_test = X_test.reshape(X_test.shape[0], 1)

model1 = LinearRegression()
visualizer = PredictionError(model1)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_train, y_train)  #valuate the model on the test data
visualizer.show()
#Bước 8 Đánh giá mô hình
print('Coefficients: ', model1.coef_)
print('Score_train: {}'.format(model1.score(X_train, y_train)))
print('Score_test: {}'.format(model1.score(X_test, y_test)))
# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(model1.predict(X_train), model1.predict(X_train) - y_train, color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(model1.predict(X_test), model1.predict(X_test) - y_test, color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
'''
Nhận Xét: Biến H6 có độ tương quan với H5 lên đến 0.752959 nhưng khi đưa vào mô hình dự đoán thì 
nhận được score_train là 0.583 và score_test là 0.468 mô hình vẫn chưa khả quan
'''
 












