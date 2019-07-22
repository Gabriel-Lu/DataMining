'''
Function: 把result.csv的汉字表示的数字转化为阿拉伯数字存储到prisonFile.csv
'''

import csv
import numpy as np

CNNum={'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,}
keyNum=CNNum.keys()


prisonFile=open("prisonFile.csv","a")
writer=csv.writer(prisonFile)

csvFile = open("result.csv", "r")
reader = csv.reader(csvFile)

for item in reader:
    #print(item)#Testing
    for i in range(len(item)):
        if(item[i] in keyNum):
            #向prisonFile文件写入中文转化后的阿拉伯数字
            writer.writerow([CNNum.get(str(item[i]))])
        else:
            writer.writerow([item[i]])
#关闭再打开prisonFile,从而可以从文件头部开始读取
prisonFile.close()
csvFile.close()



