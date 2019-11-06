# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:55:05 2019

@author: Strange
"""
#离散型实现
import numpy as np
from collections import  Counter
from sklearn.model_selection import train_test_split
car=np.loadtxt(open('D:\\car.csv'),delimiter=',',skiprows = 1, encoding='utf8',dtype=str)
car1=np.loadtxt(open('D:\\car.csv'),delimiter=',', encoding='utf8',dtype=str)
FeatureNum=car.shape[1]-1 #特征数
TotalNum=len(car) #数据的总数
feature=car1[0][0:-2]#特征集
car,test_data=train_test_split(car,test_size=0.10)#划分为数据集和测试集
#print(feature)
#根据类别对数据分类
def separatedata(dataset):
    separated={}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
data=separatedata(car)
Cnum=len(data) #类别数
#print(Cnum)

#计算类先验概率()
def calPriProb(dataset):  
    eachkindnum=[] #储存每一类的数量
    eachkindprob=[] #储存每一类的概率
    datafake=dataset.copy()
    for i in datafake.values(): #获取每一类的数量
        num=len(i)
        eachkindnum.append(num)
    for j in range(len(eachkindnum)): #获取每一类得概率
        eachkindprob.append((eachkindnum[j]+1)/(TotalNum+Cnum))
    i=0
    for kind in datafake:
        datafake[kind]=eachkindprob[i]
        i=i+1
    #print(datafake)
    #print(eachkindnum)
    return datafake,eachkindnum  #返回类条件概率，及每一类的数量
everyKindPro,eachkindnum=calPriProb(data)
def kindfeature(dataset):#统计每一类的属性数量,方便拉普拉斯平滑运算
    eachfeanum=[]
    for i in range(FeatureNum):
        ss=len(Counter(dataset[:,i]))
        eachfeanum.append(ss)
    #print(eachfeanum)
    return eachfeanum
eachfea_num=kindfeature(car)        
 #计算类条件概率
#统计数据 
alldata=[] #储存每一类的数据
for i in data.values():      
    alldata.append(i)
def address1(dataset):#统计某一类所有特征的不同情况
    result=[]    
    datasets=np.array(dataset)
    for i in range(FeatureNum):
        result.append(Counter(datasets[:,i]))   
    return result

def adress2(dataset):#计算每一特征的各种情况的条件概率
    for i in range(FeatureNum):
        vector=dataset[i]
        totalnum=0
        for key in vector:
            totalnum+=vector[key]
        for key in vector:
            vector[key]=(vector[key]+1)/(totalnum+len(vector))    
    return(dataset)
def adress3(dataset): #合并adress1 adress2
    result1=address1(dataset)
    lastresult=adress2(result1)
    return lastresult

def store(dataset):#对所有类别所有特征的情况进行储存
    names=locals()
    i=0
    while(i<Cnum):#将所有类的条件概率        
        names['statistic%s'%i]=adress3(dataset[i])
        #print(names['statistic%s'%i])
        i=i+1
    allstatistic=[]
    for i in range(Cnum):
        allstatistic.append(names['statistic%s'%i]) 
    #print(allstatistic)
    return allstatistic
allstatistic=store(alldata)
def predict_single(dataset,eachkind_num,eachfeanum,single): #对单一情况的预测    
    #有四类情况，需要分别对四种情况下的类别进行计算
    datasets=dataset#由于之前添加到list里面故先进入一层
    four_pro=[]#储存四个标签所对应概率
    i=0
    for key in everyKindPro:
        the_end_pro=everyKindPro[key]                 
        k=0          
        for j in datasets[i]:#每一类的特征数据 
            if single[k] not in j:#如果某个类的某个特征取值并未在训练集上出现，为了避免出现0的情况，分子取1(即lamda平滑因子，取1时为拉普拉斯平滑)
                the_end_pro=the_end_pro*(1/(eachkind_num[i]+eachfeanum[k]))
            else:
                the_end_pro=the_end_pro*j[single[k]]
            k=k+1
        #print(the_end_pro)
        four_pro.append(the_end_pro)           
        i=i+1
    #print(four_pro)
    lastkindpro=everyKindPro.copy()
    m=0
    for key in lastkindpro:
        lastkindpro[key]=four_pro[m]
        m=m+1
    #print(lastkindpro)
    return lastkindpro

#对测试集的预测
def test_train(dataset):
    correct=0#记录预测正确的数量
    for i in range(len(dataset)):
        result=predict_single(allstatistic,eachkindnum,eachfea_num,dataset[i][0:-1])
        for key,value in result.items():
           if(value == max(result.values())):
               prob=key
        if(prob==dataset[i][-1]):
            correct=correct+1
    acc=correct/(len(dataset))
    return acc
lastacc=test_train(test_data)
print(lastacc)