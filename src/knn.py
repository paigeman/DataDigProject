# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:04:57 2020

@author: poizi
"""

# 导包
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics.pairwise import euclidean_distances
import heapq
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# 数据
train_df = pd.read_csv("../data/train_2.csv")
test_df = pd.read_csv("../data/test2.csv")

# 特征列(不包括id和price)
column_feature = train_df.columns[1:-1]

# 训练数据
train_array = np.array(train_df[column_feature])

def knn_classify(data,k):
    """
    

    Parameters
    ----------
    data : 测试样本数据
        DESCRIPTION.
    k : 最近邻数目
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # 列数
    #lenx = len(column_feature)
    # 行数
    leny = len(train_df)
    # 存放距离的列表
    distance = []
    # 存放距离最小的类别
    class_list = []
    # 类列
    class_column = train_df["price"]
    # 多数类标号
    max_class_no = 0
    # 多数类出现的次数
    max_class_count = 0
    # 临时计数
    class_count = 0
    # reshape/符合余弦距离计算要求？
    #data = list(data)
    for i in range(leny):
        temp = train_array[i]
        # reshape/符合余弦距离计算要求？
        #temp = list(temp)
        temp = temp.reshape(1,-1)
        #print(temp)
        #print(data)
        # 可以更换其它距离
        # 余弦距离 k=5最佳
        dist = paired_distances(temp, data, metric='cosine')
        # 欧式 k=5
        #dist = euclidean_distances(temp,data)
        # 马式
        #X = np.vstack([temp, data])
        #XT = X.T
        #dist = pdist(XT, 'mahalanobis')
        # reshape/符合余弦距离计算要求？
        #print(dist)
        distance.insert(i, dist)
        pass
    # 找出距离最小/与样本最近的k个的索引
    result = list(map(distance.index,heapq.nsmallest(k, distance)))
    #print(heapq.nsmallest(k, distance))
    #print(result)
    for i in range(len(result)):
        class_list.insert(i, class_column[result[i]])
        pass
    # 可加距离权值改进 改了之后莫名更低 k=2最优
    #print(class_list)
    """
    for item in class_column:
        for i in range(len(class_list)):
            if item==class_list[i]:
                class_count += 1 / pow(distance[result[i]], 2)
                pass
            pass
        if class_count>max_class_count:
            max_class_no = item
            max_class_count = class_count
            pass
    pass
    """
    for item in class_list:
        class_count = class_list.count(item)
        if class_count > max_class_count:
            max_class_no = item
            max_class_count = class_count
            pass
        pass
    
    return max_class_no
    pass

def predict(data,k):
    """
    预测单个样本类别

    Parameters
    ----------
    data : 测试样本数据
        DESCRIPTION.

    Returns
    -------
    None.

    """
    class_no = knn_classify(data, k)
    print("该样本的价格分档是%d"%class_no)
    pass

def calculate_accuracy(data,k):
    """
    计算准确率

    Parameters
    ----------
    data : 测试集
        DESCRIPTION.
    k : 最近邻数目
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #test_array = np.array(data[column_feature])
    
    true_count = 0
    length = len(data)
    for i in range(length):
        sample = data[column_feature][i:i+1]
        test_array = np.array(sample)
        predict_classNo = knn_classify(test_array, k)
        actual_classNo = list(data["price"])[i]
        #print("predict_classNo%d"%predict_classNo,"actual_classNo%d"%actual_classNo)
        if predict_classNo==actual_classNo:
            true_count += 1
            pass
        pass
    accuracy = true_count / length
    return accuracy
    pass

def pack_knn_classify(data,k):
    """
    使用sklearn包knn模型

    Parameters
    ----------
    data : 单个样本数据
        DESCRIPTION.
    k : 最近邻个数
        DESCRIPTION.

    Returns
    -------
    None.

    """
    classify = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    classify.fit(train_array,np.array(train_df["price"]))
    class_no = classify.predict(data)
    return class_no
    pass

def pack_predict(data,k):
    class_no = pack_knn_classify(data, k)
    print("该样本的价格分档是%d"%class_no)
    pass

def pack_calculate_accuracy(data,k):
    """
    

    Parameters
    ----------
    data : 测试集
        DESCRIPTION.
    k : 最近邻数目
        DESCRIPTION.

    Returns
    -------
    None.

    """
    true_count = 0
    length = len(data)
    #print(length)
    for i in range(length):
        sample = data[column_feature][i:i+1]
        test_array = np.array(sample)
        predict_classNo = pack_knn_classify(test_array, k)
        actual_classNo = list(data["price"])[i]
        #print("predict_classNo%d"%predict_classNo,"actual_classNo%d"%actual_classNo)
        if predict_classNo==actual_classNo:
            true_count += 1
            pass
        pass
    accuracy = true_count / length
    return accuracy
    pass

def pack_calculate_accuracy_1(data,k):
    """
    sklearn自带评价

    Parameters
    ----------
    data : 测试集
        DESCRIPTION.
    k : 最近邻个数
        DESCRIPTION.

    Returns
    -------
    accuracy : TYPE
        DESCRIPTION.

    """
    classify = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)
    classify.fit(train_array,np.array(train_df["price"]))
    accuracy = classify.score(np.array(data[column_feature]),np.array(data["price"]))
    return accuracy
    pass

"""
def showFigure(data,k):
    fig = plt.figure()   
    pass
"""

"""
main
"""
#sample = test_df[0:1]
#del sample["id"]
#del sample["price"]
#sample = sample.reshape()
#data = np.array(sample)
# k=7 right
#pack_predict(data,7)
accuracy = pack_calculate_accuracy(test_df,3)
#accuracy = calculate_accuracy(test_df,3)
print("该模型的准确率是%f"%accuracy)
total_df = pd.concat([train_df,test_df])
classify = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
#classify.fit(train_array,np.array(train_df["price"]))
score_cv = cross_val_score(classify,total_df[total_df.columns[1:-1]],total_df["price"]).mean()
print("sklearn 交叉验证准确率为%f"%score_cv)