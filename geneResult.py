'''
Time:2019/07/22/19:56
Author:Lulu
Function:把 被害人BN 死亡BN当庭BN。。。格式的文本转化成：有期徒刑x年 缓刑y年的格式的txt
'''
import csv

#读取OrderFilter.csv文件
filename='/home/tabrielluunn/PycharmProjects/untitled/OrderFilter.csv'

#把事件结果因素写入到文件result.csv中
rFile='/home/tabrielluunn/PycharmProjects/untitled/result.txt'

with open(filename) as f:
    reader=csv.reader(f)
    for row in reader:
        if(row[-1]!='O'): #如果该行不是标识案例间分割线的1,2,3...
            if(row[-1] == "BR"):
                output= open(rFile,'a')
                print(row[-2])
                output.write(row[-2])
                output.close()

            if (row[-1] == 'IR'):
                output=open(rFile,'a')

                #with open(rFile, 'a') as rf:
                print(row[-2])
                output.write(row[-2])
                output.close()

            if (row[-1] == 'ER'):
                output= open(rFile, 'a')
                output.write(row[-2])#输出为：一年；二；
                output.write('\n')
                output.close()
        else:
            output= open(rFile, 'a')
            output.write('EmptyOrEnd\n')
            output.close()

'''
下一步：
'''

