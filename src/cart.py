# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:30:49 2020

@author: poizi
"""

import pandas as pd
#import time
import queue
import numpy as np
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
import pydotplus
from sklearn.model_selection import cross_val_score

# 数据
train_df = pd.read_csv("../data/train_2.csv")
test_df = pd.read_csv("../data/test2.csv")


class Node:
    
    """
    决策树结点类型
    """
    
    def __init__(self,data=None,left=None,right=None,feature=None,split=None,value=None):
        """
        初始化

        Parameters
        ----------
        data : 结点包含的数据
            结点包含的数据.
        left : 左孩子
            左孩子.
        right : 右孩子
            右孩子.
        feature : 划分特征
            划分特征.
        split : 划分点
            划分点.
        value : 属于哪类(叶子结点专有)
            属于哪类(叶子结点专有).

        Returns
        -------
        None.

        """
        self.data = data
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.value = value
        pass
    
class CartTree:
    
    """
    Cart决策树
    """
    
    def __init__(self,train,features,column_continuous):
        """
        初始化

        Parameters
        ----------
        train : 训练集
            训练集.
        features : 特征
            特征.
        label : 类标签
            类标签.

        Returns
        -------
        None.

        """
        self.train = train
        self.features = features
        #self.label = label
        self.column_continuous = column_continuous
        # 根节点
        self.root = Node()
        self.create_tree()
        pass
    
    def create_tree(self):
        """
        生成cart决策树

        Returns
        -------
        None.

        """
        # 用于存放每层的树结点
        q = queue.Queue()
        # 存放各特征ΔG的list
        delta_G_list = []
        # 存放连续特征划分点的list
        split_list = []
        for i in range(len(self.features)):
            # delta_G
            delta_G = 0.0
            # 划分点
            # 连续属性专用
            split = 0
            if self.features[i] in self.column_continuous:
                delta_G,split = self.calc_Delta_G(self.train,self.features[i])
                pass
            else:
                delta_G = self.calc_Delta_G(self.train,self.features[i])
                pass
            delta_G_list.insert(i,delta_G)
            split_list.insert(i,split)
            pass
        # 把具有最大差异性损失赋值的特征名给根节点的feature属性
        self.root.feature = self.features[delta_G_list.index(max(delta_G_list))]
        # 划分点 连续属性为split 离散属性为0
        self.root.split = split_list[delta_G_list.index(max(delta_G_list))]
        # 根结点的data为数据集
        self.root.data = self.train
        q.put(self.root)
        while not q.empty():
            node = q.get()
            #print(node.feature)
            #unique_class = set(node.data["price"])
            # 只有一个类别
            """
            if node.data.empty:
                # 为叶子结点 继续下一次循环
                continue
                pass
            """
            childL = Node(data=pd.DataFrame())
            childR = Node(data=pd.DataFrame())
            if node.feature in self.column_continuous:
                dataL = self.split_data_continuous(node.data,node.feature,node.split,0)
                dataR = self.split_data_continuous(node.data,node.feature,node.split,1)
            else:
                unique_value = set(node.data[node.feature])
                dataL = self.split_data_discrete(node.data,node.feature,unique_value[0])
                dataR = self.split_data_discrete(node.data,node.feature,unique_value[1])
                # 去除划分完毕的特征
                del dataL[node.feature]
                del dataR[node.feature]
                pass #else
            if len(set(dataL["price"]))!=1:
                # 左孩子
                childL.data=dataL
                pass #if
            else:
                childL.value = list(dataL["price"])[0]
                pass
            if len(set(dataR["price"]))!=1:
                # 右孩子
                childR.data=dataR
                pass #if
            else:
                childR.value = list(dataR["price"])[0]
                pass
            for item in [childL,childR]:
                if item.data.empty:
                    continue
                    pass #if
                delta_G_list.clear()
                split_list.clear()
                for i in range(len(self.features)):
                    # delta_G
                    delta_G = 0.0
                    # 划分点
                    # 连续属性专用
                    split = 0
                    if self.features[i] in self.column_continuous:
                        delta_G,split = self.calc_Delta_G(item.data,self.features[i])
                        pass #if
                    else:
                        delta_G = self.calc_Delta_G(item.data,self.features[i])
                        pass #else
                    delta_G_list.insert(i,delta_G)
                    split_list.insert(i,split)
                    pass #for
                item.feature = self.features[delta_G_list.index(max(delta_G_list))]
                item.split = split_list[delta_G_list.index(max(delta_G_list))]
                pass #for
            node.left = childL
            node.right = childR
            # 把左右子女进队
            # 叶子结点不进队
            if not childL.data.empty:
                q.put(childL)
                pass
            if not childR.data.empty:
                q.put(childR)
                pass
            """
            if childL!=None:
                q.put(childL)
                pass #if
            if childR!=None:
                q.put(childR)    
                pass #if
            """
            pass #while
        pass #def
    
    def calc_Gini(self,data):
        """
        计算G(t)
        求和时未计算所有类别
        因为即使算上去，划分后的数据无这个类别，也是0
        Parameters
        ----------
        data : DataFrame
            被结点划分后的数据？.

        Returns
        -------
        None.

        """
        # 数据集长度
        length = len(data)
        # 存放各个类别出现次数
        class_length = list(data.groupby("price")["price"].count())
        # gini
        gini = 1.0
        for i in range(len(class_length)):
            gini -= class_length[i] / length
            pass
        return gini
        pass
    
    def calc_Delta_G(self,data,feature):
        #print(feature)
        # delta_G
        delta_G = 0.0
        # 如果是连续特征的话
        if feature in self.column_continuous:
            # 该连续特征的列
            column = list(data[feature])
            # 排序
            column.sort()
            # 存放相邻值中点 100来个算
            list_length = 0
            if int(len(column)%200) != 0:
                list_length = int(len(column)/200)+1
                pass
            else:
                list_length = int(len(column)/200)
                pass
            mid_point_list = list(range(list_length))
            for i in range(len(mid_point_list)-1):
                temp = column[i*200:(i+1)*200]
                mid_point_list[i] = np.mean(temp)
                pass
            # 首尾一起开始计算 减短时间
            # zip构成元素对
            # i从头开始
            # j从倒数第二个开始
            # i+1 = j 停止
            """
            # 时间复杂度过高
            for i,j in zip(range(len(column)),range(len(column)-2,-1,-1)):
                mid_point_list[i]=(column[i] + column[i+1])/2
                mid_point_list[j]=(column[j] + column[j+1])/2
                if (i+1)==j:
                    break
                    pass
                pass
            """
            # 存放差异性损失
            #print(len(column))
            delta_G_list = list(range(len(mid_point_list)))
            base_gini = self.calc_Gini(data)
            for i in range(len(mid_point_list)):
                temp_delta_G = base_gini
                for k in range(2):
                    sub = self.split_data_continuous(data,feature,mid_point_list[i],k)
                    sub_gini = self.calc_Gini(sub)
                    percent = len(sub) / len(data)
                    temp_delta_G -= percent*sub_gini
                    pass
                delta_G_list[i]=temp_delta_G
                pass
            """
            for i,j in zip(range(len(column)),range(len(column)-1,-1,-1)):
                temp_delta_G_i = base_gini
                temp_delta_G_j = base_gini
                # 大于的分支 小于的分支
                for k in range(2):
                    sub_i = self.split_data_continuous(data,feature,mid_point_list[i],k)
                    sub_j = self.split_data_continuous(data,feature,mid_point_list[j],k)
                    sub_gini_i = self.calc_Gini(sub_i)
                    sub_gini_j = self.calc_Gini(sub_j)
                    percent_i = len(sub_i) / len(data)
                    percent_j = len(sub_j) / len(data)
                    temp_delta_G_i -= percent_i*sub_gini_i
                    temp_delta_G_j -= percent_j*sub_gini_j
                    pass
                delta_G_list[i]=temp_delta_G_i
                delta_G_list[j]=temp_delta_G_j
                if (i+1)==j:
                    break
                    pass
                pass
            """
            # 取最大差异性损失
            #print(delta_G_list)
            #print(feature)
            delta_G = max(delta_G_list)
            # 取第一个差异性损失最大的相邻值中点
            split = mid_point_list[delta_G_list.index(delta_G)]
            return delta_G,split
            pass
        else:
            # 未考虑构成超类
            # 非连续特征取值的可能
            unique_value = set(data[feature])
            # G(r)
            base_gini = self.calc_Gini(data)
            delta_G = base_gini
            # 各分支计数
            num_sub_branch = data.groupby(feature)[feature].count()
            for value in unique_value:
                sub = self.split_data_discrete(data,feature,value)
                # 分支G(r)
                sub_gini = self.calc_Gini(sub)
                # SL/(SL+SR) 或 SR/(SL+SR)
                # num_sub_branch[value] == len(sub)
                percent = num_sub_branch[value] / len(data)
                delta_G -= percent*sub_gini
                pass
            return delta_G
            pass
        pass
    
    def split_data_discrete(self,data:pd.DataFrame,feature,value):
        """
        划分数据框
        离散型
        Parameters
        ----------
        data : pd.DataFrame
            pandas的DataFrame.
        feature : 字符串
            特征.
        value : 数值
            特征取值.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return data[data[feature]==value]
        pass
    
    def split_data_continuous(self,data:pd.DataFrame,feature,value,choice):
        """
        划分数据框
        连续型
        Parameters
        ----------
        data : pd.DataFrame
            pandas的DataFrame.
        feature : 字符串
            特征.
        value : 数值
            特征取值.
        choice : 数值
            0表示小于等于
            1表示大于.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if choice==0:
            return data[data[feature]<=value]
            pass
        else:
            return data[data[feature]>value]
            pass
        pass
    
"""
main
"""
"""
column_continuous = ['feat1','feat7','feat8','feat9','feat10','feat11',
                     'feat13','feat14','feat15','feat17']
start_time = time.time()
cartTree = CartTree(train_df,train_df.columns[1:-1],column_continuous)  
end_time = time.time()
print("耗时%f"%(end_time-start_time))
"""
# 封装的包实现
# 指定参数 max_depth=len(train_df.columns[1:-1]) max_features=len(train_df.columns[1:-1])
classify = DecisionTreeClassifier(max_depth=4)
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
graph.write_pdf("../img/cart.pdf")
graph.write_png("../img/cart.png")
test_x = test_df[test_df.columns[1:-1]]
test_y = test_df["price"]
accuracy = classify.score(test_x,test_y)
print("该模型的准确率为%f"%accuracy)
# 剪枝 希望得到有一定分类精度且复杂程度也一般的树
# sklearn 无法实现后剪枝 可以实现前剪枝
total_df = pd.concat([train_df,test_df])
score_cv = cross_val_score(classify,total_df[total_df.columns[1:-1]],total_df["price"]).mean()
print("sklearn 交叉验证准确率为%f"%score_cv)
