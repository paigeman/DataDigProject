# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:02:01 2020

@author: poizi
"""
# 导入numpy
import numpy as np
# 导入pandas
import pandas as pd
# 导入matplotlib
import matplotlib.pyplot as plt
# 导入pyplot中所有图形对象
import plotly.graph_objs as go
# 导入plotly的一系列包
import chart_studio.plotly as py
import chart_studio
from chart_studio import tools
chart_studio.tools.set_credentials_file(username='', api_key='')

# 读取数据文件
df = pd.read_csv("../data/train.csv")
feature = input("请输入离散特征：")
if feature=="feat2":
    # 直方图对象1
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "无蓝牙",
        opacity = 0.75
    )
    # 直方图对象2
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有蓝牙",
        opacity = 0.75
    )
    title = "是否有蓝牙与价格分档的关系"        
elif feature=="feat4":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "无双SIM卡支持",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有双SIM卡支持",
        opacity = 0.75
    )
    title = "有双SIM卡支持与否与价格分档的关系"
elif feature=="feat6":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "不支持4G",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有支持4G",
        opacity = 0.75
    )
    title = "是否支持4G与价格分档的关系"
elif feature=="feat10":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "处理器核心数为1",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==2]["price"],
        name = "处理器核心数为2",
        opacity = 0.75
    )
    hist3 = go.Histogram(
        x = df.loc[df[feature]==3]["price"],
        name = "处理器核心数为3",
        opacity = 0.75
    )
    hist4 = go.Histogram(
        x = df.loc[df[feature]==4]["price"],
        name = "处理器核心数为4",
        opacity = 0.75
    )
    hist5 = go.Histogram(
        x = df.loc[df[feature]==5]["price"],
        name = "处理器核心数为5",
        opacity = 0.75
    )
    hist6 = go.Histogram(
        x = df.loc[df[feature]==6]["price"],
        name = "处理器核心数为6",
        opacity = 0.75
    )
    hist7 = go.Histogram(
        x = df.loc[df[feature]==7]["price"],
        name = "处理器核心数为7",
        opacity = 0.75
    )
    hist8 = go.Histogram(
        x = df.loc[df[feature]==8]["price"],
        name = "处理器核心数为8",
        opacity = 0.75
    )
    """
    hist9 = go.Histogram(
        x = df.loc[df[feature]==9]["price"],
        name = "处理器核心数为9",
        opacity = 0.75
    )
    """
    title = "处理器核心与价格分档的关系"
elif feature=="feat18":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "无3G",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有3G",
        opacity = 0.75
    )
    title = "是否有3G与价格分档的关系"
elif feature=="feat19":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "无触摸屏",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有触摸屏",
        opacity = 0.75
    )
    title = "是否有触摸屏与价格分档的关系"
elif feature=="feat20":
    hist1 = go.Histogram(
        x = df.loc[df[feature]==0]["price"],
        name = "无wifi",
        opacity = 0.75
    )
    hist2 = go.Histogram(
        x = df.loc[df[feature]==1]["price"],
        name = "有wifi",
        opacity = 0.75
    )
    title = "是否有wifi与价格分档的关系"
if feature=="feat10":
    data = [hist1,hist2,hist3,hist4,hist5,hist6,hist7,hist8]
    layout = go.Layout(title=title,xaxis=dict(title='price'))
else :
    data = [hist1,hist2]
    layout = go.Layout(barmode="overlay",title=title,xaxis=dict(title='price'))
fig = go.Figure(data=data,layout=layout)
py.iplot(fig,auto_open=True)
print(df.groupby(feature)["price"].describe())



