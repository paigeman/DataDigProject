# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 12:02:01 2020

@author: poizi
"""
# 导包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("../data/train.csv")

plt.figure(figsize=(16,10), dpi= 80)
feature = input("请输入连续特征：")
if feature=="feat1":
    sns.kdeplot(df["feat1"],df["price"],shade=True,color="g",label="feat1")
    plt.title('电池可以一次性存储的总能量与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat3":
    sns.kdeplot(df["feat3"],df["price"],shade=True,color="g",label="feat3")
    plt.title('微处理器执行指令的速度与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat5":
    sns.kdeplot(df["feat5"],df["price"],shade=True,color="g",label="feat5")
    plt.title('前置摄像头百万像素与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat7":
    sns.kdeplot(df["feat7"],df["price"],shade=True,color="g",label="feat7")
    plt.title('内存（以GB 为单位）与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat8":
    sns.kdeplot(df["feat8"],df["price"],shade=True,color="g",label="feat8")
    plt.title('移动深度（cm）与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat9":
    sns.kdeplot(df["feat9"],df["price"],shade=True,color="g",label="feat9")
    plt.title('重量与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat11":
    sns.kdeplot(df["feat11"],df["price"],shade=True,color="g",label="feat11")
    plt.title('主要相机百万像素与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat12":
    sns.kdeplot(df["feat12"],df["price"],shade=True,color="g",label="feat12")
    plt.title('像素分辨率高度与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat13":
    sns.kdeplot(df["feat13"],df["price"],shade=True,color="g",label="feat13")
    plt.title('像素分辨率宽度与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat14":
    sns.kdeplot(df["feat14"],df["price"],shade=True,color="g",label="feat14")
    plt.title('以兆字节为单位的随机存取存储器与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat15":
    sns.kdeplot(df["feat15"],df["price"],shade=True,color="g",label="feat15")
    plt.title('屏幕高度（以 cm 为单位）与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat16":
    sns.kdeplot(df["feat16"],df["price"],shade=True,color="g",label="feat16")
    plt.title('屏幕宽度（以 cm 为单位）与价格关系的概率密度曲线', fontsize=22)
elif feature=="feat17":
    sns.kdeplot(df["feat17"],df["price"],shade=True,color="g",label="feat17")
    plt.title('单个电池充电时间最长的时间与价格关系的概率密度曲线', fontsize=22)
plt.legend()
plt.show()