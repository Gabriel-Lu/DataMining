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
with open("prisonFile.csv","r") as pf:
    print(pf.readlines())


csvFile.close()



