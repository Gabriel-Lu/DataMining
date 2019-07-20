import csv
import numpy as np

CNNum={'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,}
keyNum=CNNum.keys()

prisonList=['有', '期', '徒', '刑']

prisonFile=open("prisonFile.csv","a")
writer=csv.writer(prisonFile)

csvFile = open("result.csv", "r")
reader = csv.reader(csvFile)

for item in reader:
    #print(item)#Testing
    for i in range(len(item)):
        #print(item[i])#Testing
        if(item[i] in keyNum):
            #向prisonFile文件写入中文转化后的阿拉伯数字
            #print(CNNum.get(str(item[i])))#Testing
            writer.writerow([CNNum.get(str(item[i]))])
        else:
            writer.writerow([item[i]])
#关闭再打开prisonFile,从而可以从文件头部开始读取
prisonFile.close()
'''
打开“待删除换行符的已转化了的（1）.txt"，每个事件之间按照EEENNNDDD分开的
用正则表达式
若匹配到”有期徒刑x年y月“,则向该一维数组中插入(x*12+y)个月;
x,y的缺省值为0
否则，count++
至此，有期徒刑的一维数组完全形成
；;;缓刑的年限同理
'''



csvFile.close()



