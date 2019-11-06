# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:52:08 2019

@author: Strange
"""
import numpy as np
from sklearn.model_selection import train_test_split
data=np.loadtxt(open('D:\\diabetes.csv'),delimiter=',',skiprows = 1, encoding='utf8',dtype=float)

#计算平均数，标准差
def getMeanstd(data1):
    feature_num=len(data1[0])-1
    names=locals()#定义动态变量
    c_mean=[]
    c_std=[]
    for i in range (feature_num):
        names['mean%s'%i]=[]
        for j in data1:
            names['mean%s'%i].append(j[i])
    for i in range(feature_num):
        c_mean.append(np.mean(names['mean%s'%i]))
        c_std.append(np.std(names['mean%s'%i]))
    return c_mean,c_std,feature_num

#计算高斯概率密度函数
def Gauss(x,mean,stdev):
    exponent = np.exp(-(np.power(x-mean,2))/(2*np.power(stdev,2)))
    GaussProb = (1/(np.sqrt(2*np.pi)*stdev))*exponent
    return GaussProb



#计算连续型数据所属类的概率
def kindProb(test_data_tip,mean,std):
    kindspro=1
    for i in range(0,8):
        kindspro=kindspro*Gauss(test_data_tip[i],mean[i],std[i])
    return kindspro

#获取单个样本的预测概率
def predict(train_data,test_data):
    train_mean,train_std,train_feature_num=getMeanstd(train_data)
    test_probility=kindProb(test_data,train_mean,train_std)
    return test_probility

#算法实现
kind1=[]                    #储存类别一
kind2=[]                    #储存类别二
train_data,test_data=train_test_split(data,test_size=0.10)#划分为数据集和测试集
for i  in  range(len(train_data)):#数据分类
    tip=train_data[i]
    if tip[-1]==1.0:
        kind1.append(train_data[i])
    else:
        kind2.append(train_data[i])
#m1,s1,fea1=getMeanstd(kind1)  #获取第一类的均值方差
#m2,s2,fea2=getMeanstd(kind2) #获取第二类的均值方差
correct_count=0
for i in test_data:
     #预测正确的数量
    probility1=predict(kind1,i)
    probility2=predict(kind2,i)
    #print(probility1>probility2)
    if(probility1>probility2):
        bestlabel=1
        if (bestlabel==i[-1]):
            correct_count=correct_count+1
    else:
        bestlabel=0
        if(bestlabel==i[-1]):
            correct_count=correct_count+1
acc=(correct_count/float(len(test_data)))*100.0
print(acc)



