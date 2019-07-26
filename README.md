# DataMining
## 输入raw500.csv
- 内容是{词语，词性}格式的
## 按照词性对词语进行分类
> Delete0.py
- 词性中B，I，E代表begin,mid,end 。N和P代表negative 和positive
- 按照积极positive和消极negative和result对事件进行分类，分成3类
> generateResult.py 
- 把结果result中的BR，IR，ER连接起来，写入到result.txt文件
## Bool型事件的抽取
> NegMetrix.py
- 建立一个大的一维数组，每个数值代表一个案例在该要素的bol值。若词语中出现了那个词、则为1.否则为0
## 数值型事件的抽取
- 由于预测缓刑/有期徒刑时间以月份为单位，而文本中时间的表示形式是：x年y个月。所以先抽取出年份、再抽取出月份，然后相加
> prisonYearAbs.py 
- 年份抽取到一维数组
> prisonMonthAbs.py 
- 把月份抽取到一维数组。
- prisonMonthAbs其实抽取出来的是x年y.我们只取字符串最后一个字符y即可

## 汉字数字表示转化成阿拉伯数字表示： 字典
![Alt Text](https://github.com/Gabriel-Lu/DataMining/blob/master/1 )
![Alt Text](https://github.com/Gabriel-Lu/DataMining/blob/master/2 )
![Alt Text](https://github.com/Gabriel-Lu/DataMining/blob/master/3)

## 总结
- 收获
   - 熟悉回顾了python的语法
- 心得
   - NLP现在有很多的库、工具和包，而且也非常好用，做出产品其实技术实现方面并不难
   - 但是弄明白这些工具背后的原理、甚至自己制作出一个这样的工具出来还是比较难的。
   - 数据和方法同样重要，缺一不可
- 问题
   - 看了几篇论文、看不懂，且看得慢。基础知识（NLP的常识）和英语阅读能力有待提高。
   - 做应用的过程中，应该用什么样的方式方法？虽然这些事情没有标准答案，但是全靠个人摸索也不可以。理想的情况是个人摸索+指导点拨。缺乏点拨
      - sol: 先看事件抽取有关的论文、资料，若仍不能解答疑惑，则问老师
      - 灵感：做一个自动抽取ALL事件因素的工具。给定任何类型的法律裁判文本，抽取出一张大表，包括布尔型和数值型。
   - 做了事件抽取后，又开始做文本分类预测，感觉路线有些杂乱。
- 本次没有做好的地方
   - 对于github上的项目，没有看懂代码也没有搞清楚原理就开始执行。这样一来、虽然得到了结果，但是并不知道这个结果是怎么来的，没有收获，浪费时间。
   - 以后遇到类似的情况，要先搞明白原理，不能不求甚解，那样学部到东西也没有锻炼动手能力、是没有效果的。
