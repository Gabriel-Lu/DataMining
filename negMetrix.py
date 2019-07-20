import csv
import numpy as np

deathCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if(row[-1]=="死亡"):
                deathCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("DeathCount:\n")
print(deathCount)

hurtCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1]=="重伤")or(row[-1]=="受伤")or(row[-1]=="轻伤")):
                hurtCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("hurtCount:\n")
print(hurtCount)

runAwayCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1]=="逃逸")or(row[-1]=="逃离")or(row[-1]=="逃避")):
                runAwayCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("runAwayCount:\n")
print(runAwayCount)

drunkCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1]=="酒后")or(row[-1]=="醉酒")):
                drunkCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("drunkCount:\n")
print(drunkCount)

speedCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if(row[-1]=="超速"):
                speedCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("speedCount:\n")
print(speedCount)

preCount = np.zeros(652)
#读取消极事件因素文件negative.csv
nFile='/home/tabrielluunn/PycharmProjects/untitled/negative.csv'
with open(nFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if(row[-1]=="前科"):
                preCount[count]=1
                #print(row[-1])
                #print(count)
        else:
            count=count+1
print("preCount:\n")
print(preCount)

confessCount = np.zeros(652)
#读取积极事件因素文件negative.csv
pFile='/home/tabrielluunn/PycharmProjects/untitled/positive.csv'
with open(pFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1] in "认罪")or (row[-1] in "自首")or (row[-1] in "投案")):
                confessCount[count]=1
        else:
            count=count+1
print("confessCount:\n")
print(confessCount)

attCount = np.zeros(652)
#读取积极事件因素文件negative.csv
pFile='/home/tabrielluunn/PycharmProjects/untitled/positive.csv'
with open(pFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1]=="谅解")or (row[-1]=="悔罪表现")or (row[-1]=="如实")or (row[-1]=="积极")):
                attCount[count]=1
        else:
            count=count+1
print("attCount:\n")
print(attCount)

payCount = np.zeros(652)
#读取积极事件因素文件negative.csv
pFile='/home/tabrielluunn/PycharmProjects/untitled/positive.csv'
with open(pFile) as f:
    reader=csv.reader(f)
    count=0
    for row in reader:
        if(row[-1]!='EEENNNDDD'): #如果该行不是标识案例间分割线的EEENNNDDD
            if((row[-1]=="经济损失")or (row[-1] in "赔偿")):
                payCount[count]=1
        else:
            count=count+1
print("payCount:\n")
print(payCount)
