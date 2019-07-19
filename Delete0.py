#-*- coding: utf-8 -*-
# Function: Delete O type but not number row in csv file
# Author:Tabrielluunn
# Date:2019/07/19 Fri

import pandas as pd
import csv
import numpy

#生成一个500行10列的全0的矩阵
caseC=500;
featureC=10;
bigM=numpy.zeros([caseC,featureC])

#将数据导入pandas,由于这里我们有2列、生成的是dataframe(df_
df=pd.read_csv('/home/tabrielluunn/200gb/BMakeUpData/raw500.csv')#注意，CSV首行默认为标题行

#创建一个字符型的list,值是'1',‘2’,’3‘,...，‘500’
int500list=[]
int500list = range(0, 501)
str500list=[str(x)for x in int500list]#转换成字符型

OrderFilter= df[df['words'].isin(str500list)]#TESTING
#print(OrderFilter)#TESTING
#OrderFilter.type='NEXT'
#print(OrderFilter)

#OrderFilter.to_csv("OrderFilter.csv")

filter = df[(df['type'].isin(['BN','EN','BP','EP','SP','BR','IR','ER']))|(df['words'].isin(str500list))]

#print(filter.head(50))#TESTING
filter.to_csv("/home/tabrielluunn/PycharmProjects/untitled/OrderFilter.csv",index=False,header=False)
#得到了OrderFilter.csv,文件中保留了事件因素，去掉了无意义的一些词





