# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:20:30 2020

@author: poizi
"""

#import numpy as np
import pandas as pd
#import math
from scipy.stats import norm
#from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv("../data/train_2.csv")
test_df = pd.read_csv("../data/test2.csv")

column_continuous = ['feat1','feat7','feat8','feat9','feat10','feat11',
                     'feat13','feat14','feat15','feat17']
column_feature = train_df.columns[1:-1]

unique_value = set(train_df[train_df.columns[-1]])

def calc_x_c(data):
    """
    计算样本x属于最大后验概率的类

    Parameters
    ----------
    data : 样本数据
        DESCRIPTION.
        
    Returns
    -------
    None.

    """
    max_x_c = 0.0
    max_classNo = 0
    for classNo in unique_value:
        x_c = 1
        for feature in column_feature:
            # bug?
            #print(feature)
            data_gb = train_df.groupby("price")[feature]
            mean = data_gb.mean()
            #var = data_gb.var()
            var = data_gb.std()
            count_i = data_gb.count()
            count_sik = train_df.groupby([feature,"price"])[feature].count()
            if feature in column_continuous:
                #print(data)
                x_c *= norm.pdf(x=list(data[feature])[0],loc=mean[classNo],scale=var[classNo])
                #print(norm.cdf(x=list(data[feature])[0],loc=mean[classNo],scale=var[classNo]))
                #print(x_c)
                pass
            else:
                x_c *=  count_sik[list(data[feature])[0]][classNo]/count_i[classNo] 
                #print(x_c)
                pass
            #print(x_c)
        x_c *= train_df.groupby("price")["price"].count()[classNo] / len(train_df)
        if x_c > max_x_c:
            max_x_c = x_c
            max_classNo = classNo
        #print(x_c)
        #print(max_x_c)
        #print(max_classNo)
        pass
    #print(max_classNo)
    return max_classNo
    pass

def predict(data):
    """
    预测单个样本价格分档

    Parameters
    ----------
    data : 单个测试样本数据
        DESCRIPTION.

    Returns
    -------
    None.

    """
    class_No = calc_x_c(data)
    print("该样本的价格分档是%d"%class_No)
    pass

def calculate_accuracy(data):
    """
    计算测试集分类准确率/模型准确率

    Parameters
    ----------
    data : 测试集
        DESCRIPTION.

    Returns
    -------
    None.

    """
    length = len(data)
    #print(length)
    true_count = 0
    for i in range(length):
        sample = data[i:i+1]
        #print("第一行%d"%i)
        predict_classNo = calc_x_c(sample)
        actual_classNo = list(sample["price"])[0]
        if predict_classNo==actual_classNo:
            true_count += 1
            pass
        pass
    accuracy = true_count / length
    return accuracy
    pass

"""
def fitDgree(data):
    out = {}
    for value in unique_value:
        out[value] = {}
        for feature in column_continuous:
            out[value][feature] = stats.kstest(data[data["price"]==value][feature],"norm")
            pass
        pass        
    return out
    pass
"""

"""
main
"""
#sample = test_df[4:5]
#predict(sample)
#print("该模型的准确率是%f"%(calculate_accuracy(test_df)))
#fitDgree = fitDgree(train_df)
#print(fitDgree)
classify = GaussianNB()
classify.fit(train_df[train_df.columns[1:-1]],train_df["price"])
accuracy = classify.score(test_df[test_df.columns[1:-1]],test_df["price"])
print("sklearn 贝叶斯模型的准确率为%f"%accuracy)
total_df = pd.concat([train_df,test_df])
score_cv = cross_val_score(classify,total_df[total_df.columns[1:-1]],total_df["price"]).mean()
print("sklearn 交叉验证准确率为%f"%score_cv)

