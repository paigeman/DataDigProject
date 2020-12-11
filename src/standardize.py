# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 12:02:01 2020

@author: poizi
"""

# 导包
from sklearn import preprocessing  
import pandas as pd

# 读取数据
df = pd.read_csv("../data/train_1.csv")

#建立MinMaxScaler对象
min_max_scaler = preprocessing.MinMaxScaler()
"""
print(df.iloc[:, 1:15])
"""
array_MinMax = min_max_scaler.fit_transform(df.iloc[:, 1:15])

df_MinMax = pd.DataFrame(array_MinMax)

df_MinMax.insert(0,"id",df["id"])
df_MinMax.insert(15,"price",df["price"])
df_MinMax.columns = df.columns
"""
print(df_MinMax)
"""
# 把规范化的数据写入新的文件
df_MinMax.to_csv("../data/train_2.csv",index=False)

