# -*- coding: utf-8 -*-
"""
Created on Tue Dec 1 12:02:01 2020

@author: poizi
"""

# 导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("../data/train_1.csv")
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
# square参数保证corrmat为非方阵时，图形整体输出仍为正方形
sns.heatmap(corrmat, annot=True, annot_kws={'size':8},vmax=.8,square=True,ax=ax)
plt.show()
"""
k = 10
top10_attr = corrmat.nlargest(k, "price").index
top10_mat = corrmat.loc[top10_attr, top10_attr]
fig,ax = plt.subplots(figsize=(8,6))
sns.set(font_scale=1.25)
sns.heatmap(top10_mat, annot=True, annot_kws={'size':12}, square=True)
# 设置annot使其在小格内显示数字，annot_kws调整数字格式
plt.show()
"""