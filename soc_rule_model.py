# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:34:01 2020

@author: David
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 能耗影响因素分析
df_soc_rule = pd.read_csv('D:/zhongdiao/上汽大众数据中台/需求/数据分析建模-张俊/data/pyspark_data/dbh_soc_rule/dbh_soc_rule.csv')
features = ['rush_times_per100km', 'nasty_times_per100km', 'avg_speed', 'start_soc', 'start_hour', 'trip_distance', 'delta_soc_per100km']
df = df_soc_rule[features]

# 数据查看
df['rush_times_per100km'].hist(bins=100)
plt.title('rush_times_per100km')
df['rush_times_per100km'].describe()
df['rush_times_per100km'].quantile([0.8, 0.85, 0.9, 0.95])


df['nasty_times_per100km'].hist(bins=100)
plt.title('nasty_times_per100km')
df['nasty_times_per100km'].describe()
df['nasty_times_per100km'].quantile([0.8, 0.85, 0.9, 0.95])


df['delta_soc_per100km'].hist(bins=100)
plt.title('delta_soc_per100km')
df['delta_soc_per100km'].describe()
df['delta_soc_per100km'].quantile([0.8, 0.85, 0.9, 0.95])


df['avg_speed'].hist(bins=100)
plt.title('avg_speed')
df['avg_speed'].describe()
df['avg_speed'].quantile([0.8, 0.85, 0.9, 0.95])


df['start_soc'].hist(bins=100)
plt.title('start_soc')
df['start_soc'].describe()
df['start_soc'].quantile([0.8, 0.85, 0.9, 0.95])


df['start_hour'].hist(bins=100)
plt.title('start_hour')
df['start_hour'].describe()
df['start_hour'].quantile([0.8, 0.85, 0.9, 0.95])


df['trip_distance'].hist(bins=100)
plt.title('trip_distance')
df['trip_distance'].describe()
df['trip_distance'].quantile([0.8, 0.85, 0.9, 0.95])


# 数据筛选
df = df[(df['rush_times_per100km']>0) & (df['rush_times_per100km']<200) &\
        (df['nasty_times_per100km']>0) & (df['nasty_times_per100km']<200) &\
        (df['delta_soc_per100km']>0) & (df['delta_soc_per100km']<60) & (df['delta_soc_per100km']>25) &\
        (df['trip_distance']>3) & (df['start_hour']>6)]
df.reset_index(drop=True, inplace=True)


# 构建模型
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


 
x = df[features[:6]]
y = df['delta_soc_per100km']

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# data standardization
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)


# 线性回归
lr = RidgeCV()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

# 误差
mean_absolute_error(y_test, lr_pred)

# 画图
plt.plot(np.arange(len(lr_pred[:200])), lr_pred[:200], label='predict')
plt.plot(np.arange(len(lr_pred[:200])), y_test[:200], label='true')
plt.legend()
plt.show()


# RF
rfr = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=1)
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)

# 误差
mean_absolute_error(y_test, rfr_pred)

# 画图
plt.plot(np.arange(len(rfr_pred[:200])), rfr_pred[:200], label='predict')
plt.plot(np.arange(len(rfr_pred[:200])), y_test[:200], label='true')
plt.legend()
plt.show()





