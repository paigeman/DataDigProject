# -*- coding: utf-8 -*-
"""
Created on Tue Dec 3 12:02:01 2020

@author: poizi
"""

# 导包
import pandas as pd
import math
import time
import queue
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.model_selection import cross_val_score

# 读取数据
df = pd.read_csv("../data/train_2.csv")
column_continuous = ['feat1','feat7','feat8','feat9','feat10','feat11',
                     'feat13','feat14','feat15','feat17']
"""
# 列数
cols_count = len(df.columns)
# 列名
cols_names = list(df.columns)
"""

train_df = pd.read_csv("../data/train_2.csv")
test_df = pd.read_csv("../data/test2.csv")

def calculate_entropy(data: "DataFrame")->"计算信息熵":
    """
    

    Parameters
    ----------
    data : "DataFrame"
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 获取最后一列
    columns_items = data[data.columns[-1]]
    # print(data.columns[-1])
    # 获取取值以及取值的个数
    # 创建字典
    labels = {}
    # value_counts()计数，不包括NA
    # items 返回(value, count)， 值， 个数
    for item, count in columns_items.value_counts().items():
        labels[item] = count
    # 获取元组数(行数)
    rows_count = len(columns_items)
    # 计算信息熵
    entropy = 0
    for key in labels:
        # 类别的概率
        p = labels[key] / rows_count
        entropy -= p * math.log(p,2)
    return entropy

"""
def c2d(data: "DataFrame",quarter: "几分位")->"连续特征用分位数法离散化":
    df_handle = data.copy(deep=True)
    for i in range(1,cols_count):
        for j in range(0,len(column_continuous)):
            if cols_names[i] == column_continuous[j]:
                df_handle[cols_names[i]] = pd.qcut(df_handle[cols_names[i]],quarter)
                break
    return df_handle
"""

def split_data(data: "DataFrame",feature: "列名",value: "属性列的取值")->"按照属性某个取值划分的子集":
    """
    

    Parameters
    ----------
    data : "DataFrame"
        DESCRIPTION.
    feature : "列名"
        DESCRIPTION.
    value : "属性列的取值"
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 暂时存储子集数据
    sub = []
    size = len(data)
    # 判断每行特征值是否等于value
    for i in range(size):
        if data.iloc[[i], :][feature].values[0] == value:
            temp = data.loc[i]
            sub.append(list(temp))
    # 创建DataFrame对象
    sub_df = pd.DataFrame(sub,columns=list(data.columns))
    # 删除该列(已划分)
    sub_df.drop(columns=[feature],inplace=True)
    return sub_df

def handle_continuity(data: "DataFrame")->"计算哪里的分位数最好":
    """
    

    Parameters
    ----------
    data : "DataFrame"
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 保存连续值信息
    continuous_dict = {}
    # 列数
    cols_count = len(data.columns)
    # 列名
    cols_names = list(data.columns)
    for i in range(1,cols_count-1):
        if cols_names[i] in column_continuous:
            # 最优的等分位置
            best_quarter = 0
            # 最优信息增益
            best_info_gain = 0.0
            # 未被此属性划分前的信息熵
            base_entropy = calculate_entropy(data)
            # 这里尝试的分位点是2至3
            for quarter in range(2,4):
                # 中间值，属性划分的信息熵
                feature_entropy = 0.0
                data_backup = data.copy(deep=True)
                # 分位数法划分
                data_backup[cols_names[i]] = pd.qcut(data_backup[cols_names[i]],quarter,duplicates="drop")
                # 待处理列名
                feature_name = data_backup.columns[i]
                # 去重，计算feature所有可能取值
                unique_value = set(data_backup[feature_name])
                # 计算信息增益
                for value in unique_value:
                    sub = split_data(data_backup,feature_name,value)
                    rate = len(sub) / len(data_backup)
                    feature_entropy += rate * calculate_entropy(sub)
                info_gain = base_entropy - feature_entropy
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_quarter = quarter
            # 保存最好的分位点到字典
            continuous_dict[cols_names[i]] = best_quarter
    return continuous_dict
    
    
def getBestFeat(data: "DataFrame")->"计算最优特征":
    """
    

    Parameters
    ----------
    data : "DataFrame"
        DESCRIPTION.

    Returns
    -------
    None.

    """
    """
    # 获取连续值最优划分点
    continuous_dict = handle_continuity(data)
    """
    data_backup = data.copy(deep=True)
    data_columns = list(data_backup.columns)
    """
    # 保存连续特征的列表
    continuous_list = []
    # 连续特征信息保存到列表中
    for item in continuous_dict.keys():
        continuous_list.append(item)
    """
    """
    # 对连续值进行划分
    """
    name_column = data_backup.columns
    # 未被此属性划分前的信息熵
    base_entropy = calculate_entropy(data)
    # 最大信息增益及最佳划分属性
    best_info_gain = 0.0
    best_feature = ""
    # 选择特征
    for feature in name_column[1:-1]:
        unique_value = set(data_backup[feature])
        # 每个属性的信息熵
        feature_entropy = 0.0
        for value in unique_value:
            sub = split_data(data_backup,feature,value)
            rate = len(sub) / len(data_backup)
            feature_entropy += rate * calculate_entropy(sub)
        info_gain = base_entropy - feature_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature

def createTree(data: "DataFrame")->"建立ID3决策树":
    data_columns = data.columns
    columns_value = data[data_columns[-1]]
    # 递归结束条件 只有一个类
    if len(columns_value.value_counts())==1:
        return columns_value.values[0]
    # 递归结束条件 只有两列(id和类)
    if len(data_columns)==2:
        class_dict = {}
        for item,value in columns_value.value_counts().items():
            class_dict[item] = value
        class_name = max(class_dict,key=lambda k: class_dict[k])
        return class_name
    # 上诉情况外
    # 选择最佳划分属性
    best_feature = getBestFeat(data)
    if best_feature=="":
        pass
    else:
        # 建立字典树 {属性名:{分类:分支}}
        tree = {best_feature:{}}
        unique_value = set(data[best_feature])
        for value in unique_value:
            tree[best_feature][value] = createTree(split_data(data,best_feature,value))
        return tree
        

# 主函数
# 获取连续值最优划分点
"""
data = df.copy(deep=True)
continuous_dict = handle_continuity(data)
# 保存连续特征的列表
continuous_list = []
# 连续特征信息保存到列表中
for item in continuous_dict.keys():
    continuous_list.append(item)
for i in range(len(continuous_list)):
    data[continuous_list[i]] = pd.qcut(data[continuous_list[i]],continuous_dict[continuous_list[i]])
tree = createTree(data)
print(tree)
draw.createPlot(tree)
"""

"""
# 测试区
print(df.iloc[[0], :])
print(df.iloc[[0], :]["feat18"])
print(df.iloc[[0], :]["feat18"].values[0])
     
test = df.copy()
result = split_data(test,"feat18",1)
result.to_csv("../data/train_3.csv",index=False)       
        
print(handle_continuity(df))
"""     
classify = DecisionTreeClassifier(criterion="entropy",max_depth=4)
x=train_df[train_df.columns[1:-1]]
y=train_df["price"]
classify.fit(x,y)
sample = test_df[test_df.columns[1:-1]][0:1]
sample_class_no = classify.predict(sample)
print("该样本预测的价格分档为:%d"%sample_class_no)
plt.figure(figsize=(20,20))
plot_tree(classify,filled=True)
#plt.show()
feature_names = train_df.columns[1:-1]
class_names = list(map(lambda x:str(x),train_df["price"]))
dot_data = tree.export_graphviz(classify, out_file=None,
								feature_names=feature_names,
								class_names=class_names,
								filled=True, rounded=True,
								special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_size('"450!,1000!"')
graph.set_fontsize(30)
graph.write_pdf("../img/id3.pdf")
graph.write_png("../img/id3.png")
test_x = test_df[test_df.columns[1:-1]]
test_y = test_df["price"]
accuracy = classify.score(test_x,test_y)
print("该模型的准确率为%f"%accuracy)
total_df = pd.concat([train_df,test_df])
score_cv = cross_val_score(classify,total_df[total_df.columns[1:-1]],total_df["price"]).mean()
print("sklearn 交叉验证准确率为%f"%score_cv)
        

    

    