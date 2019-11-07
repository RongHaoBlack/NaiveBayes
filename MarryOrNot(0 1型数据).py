# -*- coding: utf-8 -*-
import numpy as np
dataMat=np.loadtxt(open('D:\\MarryOrNot.csv'), delimiter=',', skiprows = 1)#读取文件
row = dataMat.shape[0]#数据行数
col = dataMat.shape[1]#数据列数 
print(row)
print(col)

#假设女生现被有一男求婚，属性为：帅、性格不好、高、不上进，求嫁与不嫁的概率

#计算先验概率
print(dataMat)
p_marry=sum(dataMat[:,4])/row  #嫁的概率
p_marry_not=1-p_marry     #不嫁的概率

#函数：计算条件概率
def computP(data,x,y):
    row=data.shape[0]
    marry=0
    marryNot=0
    yInMarry=0
    yInMarryNot=0
    for i in range(row):
        if data[i,x]==1: 
            marry=marry+1#嫁的次数
            if data[i,y]==1: 
                    yInMarry=yInMarry+1#嫁情况下，y事件发生次数
        if data[i,x]==0: 
           marryNot=marryNot+1#不嫁的次数
           if data[i,y]==1:
               yInMarryNot=yInMarryNot+1#不嫁情况下，y事件发生次数
    p_y_marry=yInMarry/marry#嫁情况下，y事件发生概率
    p_n_marry=1-p_y_marry#嫁情况下，y事件不发生概率
    p_y_marryNot=yInMarryNot/marryNot#不嫁情况下，y事件发生概率
    p_n_marryNot=1-p_y_marryNot#不嫁情况下，y事件不发生概率
    return p_y_marry,p_n_marry,p_y_marryNot,p_n_marryNot
#开始计算
if __name__ =="__main__":
    print('帅，性格不好，高，不上进')
    print('根据现有数据如何选择')
    p_shuai_marry,p_bushuai_marry,p_shuai_marryNot,p_bushuai_marryNot=computP(dataMat,4,0)
    p_x_marry,p_xbad_marry,p_x_marryNot,p_xbad_marryNot=computP(dataMat,4,1)
    p_g_marry,p_gnot_marry,p_g_marryNot,p_gnot_marryNot=computP(dataMat,4,2)
    p_s_marry,p_snot_marry,p_s_marryNot,p_snot_marryNot=computP(dataMat,4,3)
    
    #求p（嫁|帅，性格不好，高，不上进）  用贝叶斯公式
    p_t_marry=p_shuai_marry*p_xbad_marry*p_g_marry*p_snot_marry*p_marry
    p_f_marry=p_shuai_marryNot*p_xbad_marryNot*p_g_marryNot*p_snot_marryNot*p_marry_not
    print(p_t_marry)
    print(p_f_marry)
    
    pdo=p_t_marry/(p_t_marry+p_f_marry)
    pdonot=p_f_marry/(p_t_marry+p_f_marry)
    
#结果
    print("嫁的概率是：%3.2f"%pdo)
    print("不嫁的概率是：%3.2f"%pdonot)
    
   
              
           
    

