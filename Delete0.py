#-*- coding: utf-8 -*-
#function: Delete O type but not number row in csv file

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
filter.to_csv("/home/tabrielluunn/PycharmProjects/untitled/OrderFilter.csv",index=False,header=False)#得到了OrderFilter.csv,文件中保留了事件因素，去掉了无意义的一些词

filename='/home/tabrielluunn/PycharmProjects/untitled/OrderFilter.csv'
#0720 重新写一个CSV文件，合并BN，EN因为无法删除CSV文件的某一行

#把消极事件因素写入到文件negative.csv中
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(filename) as f:
    reader=csv.reader(f)
    for row in reader:
        #print(row[-1]) #TESTING
        if(row[-1]!='O'): #如果该行不是标识案例间分割线的1,2,3...
            if(row[-1] == "BN"):
                with open(nFile,'a') as nf:
                    writer=csv.writer(nf)
                    writer.writerow([row[-2]])
            if (row[-1] == 'EN'):
                with open(nFile, 'a') as nf:
                    writer = csv.writer(nf)
                    writer.writerow([row[-2]])
                    writer.writerow(',')
        else:
            with open(nFile, 'a') as nf:
                writer = csv.writer(nf)
                writer.writerow(['--------------'])

#把积极事件因素写入到文件positive.csv中
pFile='/home/tabrielluunn/PycharmProjects/untitled/positive.csv'
with open(filename) as f:
    reader=csv.reader(f)
    for row in reader:
        #print(row[-1]) #TESTING
        if(row[-1]!='O'): #如果该行不是标识案例间分割线的1,2,3...
            if(row[-1] == "BP"):
                with open(pFile,'a') as pf:
                    writer=csv.writer(pf)
                    writer.writerow([row[-2]])
            if (row[-1] == 'EP'):
                with open(pFile, 'a') as pf:
                    writer = csv.writer(pf)
                    writer.writerow([row[-2]])
                    writer.writerow(',')
            if (row[-1] == 'SP'):
                with open(pFile, 'a') as pf:
                    writer = csv.writer(pf)
                    writer.writerow([row[-2]])
        else:
            with open(pFile, 'a') as pf:
                writer = csv.writer(pf)
                writer.writerow(['--------------'])

#把事件结果因素写入到文件result.csv中
rFile='/home/tabrielluunn/PycharmProjects/untitled/result.csv'
with open(filename) as f:
    reader=csv.reader(f)
    for row in reader:
        #print(row[-1]) #TESTING
        if(row[-1]!='O'): #如果该行不是标识案例间分割线的1,2,3...
            if(row[-1] == "BR"):
                with open(rFile,'a') as rf:
                    writer=csv.writer(rf)
                    writer.writerow([row[-2]])
            if (row[-1] == 'IR'):
                with open(rFile, 'a') as rf:
                    writer = csv.writer(rf)
                    writer.writerow([row[-2]])
            if (row[-1] == 'ER'):
                with open(rFile, 'a') as rf:
                    writer = csv.writer(rf)
                    writer.writerow([row[-2]])
                    writer.writerow(',')

        else:
            with open(rFile, 'a') as rf:
                writer = csv.writer(rf)
                writer.writerow(['--------------'])


