# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:55:05 2019

@author: Strange
"""
#离散型实现
import numpy as np
from collections import  Counter
from sklearn.model_selection import train_test_split
car=np.loadtxt(open('D:\\car.csv'),delimiter=',',skiprows = 1, encoding='utf8',dtype=str)#读取数据，注意数据位于D盘
FeatureNum=car.shape[1]-1              #特征数
TotalNum=len(car)                      #数据的总数
car1,test_data=train_test_split(car,test_size=0.10)#将数据划分为car1训练集集和test_data测试集，比例为9:1
#根据类别对数据分类
def separatedata(dataset):
    separated={}                        #用于分类储存数据的字典
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):#如果该字典没有vector[-1](即标签)对应的key，则添加该key
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

data=separatedata(car1)                #data为划分完训练集的字典
Cnum=len(data)                         #类别数

#计算类先验概率()
def calPriProb(dataset):  
    eachkindnum=[]                     #储存每一类的数量
    eachkindprob=[]                    #储存每一类的概率
    datafake=dataset.copy()
    for i in datafake.values():        #获取每一类的数量
        num=len(i)
        eachkindnum.append(num)
    for j in range(len(eachkindnum)):  #获取每一类得先验概率
        eachkindprob.append((eachkindnum[j]+1)/(TotalNum+Cnum))
    i=0
    for kind in datafake:
        datafake[kind]=eachkindprob[i]  #将每一类的类先验概率作为value赋予对应的key
        i=i+1
    return datafake,eachkindnum         #返回类条件概率，及每一类的数量
everyKindPro,eachkindnum=calPriProb(data)#everyKindPro为每一类的类先验概率,eachkindnum为每一类的数量

def kindfeature(dataset):               #统计每一类特征的所对应属性的数量,方便拉普拉斯平滑运算
    eachfeanum=[]
    for i in range(FeatureNum):
        temp=len(Counter(dataset[:,i])) #Counter可以字典的形式返回属性的种类及做对应的个数，再用len()获取索引的个数
        eachfeanum.append(temp)         #将各特征的属性数量存入eachfeanum[]
    return eachfeanum
eachfea_num=kindfeature(car)        
 #计算类条件概率
#统计数据 
alldata=[]                              #储存每一类的数据
for i in data.values():      
    alldata.append(i)
def address1(dataset):                  #统计某一类所有特征的不同情况
    result=[]    
    datasets=np.array(dataset)          #便于Counter方法的应用
    for i in range(FeatureNum):
        result.append(Counter(datasets[:,i]))   
    return result

def adress2(dataset):                   #计算每一特征的各种情况的条件概率
    for i in range(FeatureNum):
        vector=dataset[i]               #每个特征
        totalnum=0
        for key in vector:
            totalnum+=vector[key]       #计算所有属性数量和
        for key in vector:
            vector[key]=(vector[key]+1)/(totalnum+len(vector))#计算类先验概率
    return(dataset)
def adress3(dataset):                   #合并adress1 adress2
    result1=address1(dataset)
    lastresult=adress2(result1)
    return lastresult

def store(dataset):                     #对所有类别所有特征的情况进行储存
    names=locals()
    i=0
    while(i<Cnum):                      #将所有类的条件概率        
        names['statistic%s'%i]=adress3(dataset[i])#计算每一类的类条件概率
        i=i+1
    allstatistic=[]
    for i in range(Cnum):
        allstatistic.append(names['statistic%s'%i])#将所有数据存入[]
    return allstatistic
allstatistic=store(alldata)             #储存所有条件概率
def predict_single(dataset,eachkind_num,eachfeanum,single): #对单一情况的预测    
    #有四类情况，需要分别对四种情况下的类别进行计算
    datasets=dataset
    four_pro=[]                         #储存四个标签所对应概率
    i=0
    for key in everyKindPro:
        the_end_pro=everyKindPro[key]   #先令最终概率为类先验概率            
        k=0                             #k的上限为 单条数据的属性数 与特征数相等 
        for j in datasets[i]:           #每一类的特征数据 
            if single[k] not in j:      #如果某个类的某个特征取值并未在训练集上出现，为了避免出现0的情况，分子取1(即lamda平滑因子，取1时为拉普拉斯平滑)
                the_end_pro=the_end_pro*(1/(eachkind_num[i]+eachfeanum[k]))
            else:                       #如果出现，则正常计算
                the_end_pro=the_end_pro*j[single[k]]
            k=k+1                       #加一则统计下一属性
        four_pro.append(the_end_pro)           
        i=i+1                           #加一为统计下一对应的类别
    lastkindpro=everyKindPro.copy()     #因为引用了全局变量，为使全局变量不改变故copy()
    m=0
    for key in lastkindpro:
        lastkindpro[key]=four_pro[m]    #将新value赋值给对应的key，因为其开始就是根据key的顺序进行计算，并将结果逐个存入list，故一一其对应顺序未改变
        m=m+1                           #key的数量与m的大小一致
    return lastkindpro
#对测试集的预测
def test_train(dataset):
    correct=0                           #记录预测正确的数量
    for i in range(len(dataset)):
        result=predict_single(allstatistic,eachkindnum,eachfea_num,dataset[i][0:-1])#每次预测一条数据
        for key,value in result.items():#挑选最大的value所对应的key
           if(value == max(result.values())):
               prob=key
        if(prob==dataset[i][-1]):       #与标签作对比
            correct=correct+1
    acc=correct/(len(dataset))
    return acc
lastacc=test_train(test_data)
print(lastacc)