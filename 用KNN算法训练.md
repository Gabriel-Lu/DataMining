# DataMinig0701
## 利用WEKA读取数据，用KNN算法训练，使得结果准确率达到100%步骤记录
   -     CMD输入java -jar weka.jar启动weka软件
   -     Preprocess中，在Filter中选择unsupervised 的Normalize，点击apply，进行归一化处理，使得几个指标的权重一样
   -     Classify中，在Classifier中选择lazy-IBK(K邻近函数），右键选择属性列。
        -     在nearsetNeighbourSearchAlgorithm中可以选择距离的种类，默认的是linearNNSearch（欧氏距离），设置KNN（K的值）
        -     采用交叉验证的方式选取K值，设置成我们向考虑最大的K值。并设置cross-validate(交叉验证）为true.
        -    K折交叉验证(K-fold cross validation)指的是把训练数据D 分为 K份，用其中的(K-1)份训练模型，把剩余的1份数据用于评估模型的质量。将这个过程在K份数据上依次循环，并对得到的K个评估结果进行合并，如求平均或投票。
   -     点击start运行
   -     查看classifier output中的结果
        -     detailed accuracy by class中各个字符的含义
             -     (TP) rate:True positive，被正确分类为class x的比率。
             -     (FP) rate:False positive，被错误分类为class x的比率。
             -     Precision：类型为class x的instances被正确分类为class x的比率。
