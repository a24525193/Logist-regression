# Logist-regression
Pattern Recognition Assignment 1


## 资料集

https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records


## 所使用的套件

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

## 汇入资料和资料检查
```
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print(data.head())
print(data.isnull().any()) 

```
查看前五笔资料形态

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/data_head.PNG "data_head")

检查是否有缺失值，结果并无缺失值

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/data_isnull_any.PNG "data_isnull_any")



## 热度图
检查所有特征值的相关系数，进而选择想要的特征

```
#heatmap
plt.figure(figsize=(15, 12))
feature_corr = data.corr()
hm = sns.heatmap(feature_corr, annot=True)
plt.show()
```

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/heatmap.png "heatmap")

最后选择与DEATH_EVENT相关系数最高的TIME和较好分辨的AGE，将以这两个特征做逻辑斯回归分类，来预测死亡事件是否发生




## 训练和预测

将X和Y以8:2的比例，分出训练集和测试集

```
X = data[["age", "time"]].values
y = data[["DEATH_EVENT"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=345453)
```

用逻辑斯回归训练模型
```
#training 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```
计算预测结果
```
#score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/accuracy.PNG "accuracy")

F1-score达到0.88

## 绘图

建制图表，放入特征值
```
plt.figure(figsize=(10, 8))
plt.scatter(data["time"],data["age"] ,s = 60 , c=y)
```
### 绘制决策边界
创建方法，设定最大值和最小值，并附加边缘填充
```
def plot_decision_boundary(m):

    #set max value, min value and edge filling
    x_min  = data["time"].min() - .5
    x_max  = data["time"].max() + .5

    y_min = data["age"].min() - .5
    y_max = data["age"].max() + .5
    
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
```

使用预测方法进行预测
```
    #use prediction function to predict
    Z = m(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
```
画出决策边界，并将显示图标和颜色分别标示，以便更明显的区分
```
    #plot
    plt.contourf(xx, yy, Z)

    for i in range(data.shape[0]):
        if y[i] == 1:
            died = plt.scatter(data.time[i],data.age[i],marker = "x", s = 40 , color = "green")
        
        else:
            alive = plt.scatter(data.time[i],data.age[i],marker = "o", s = 40 , color = "blue")
    
    plt.legend((died,alive),('1','0'),title= "DEATH_EVENT")
    
plot_decision_boundary(lambda x: classifier.predict(x))
```
调整一下图表显示内容，并输出图表
```
plt.xlabel('Time')  
plt.ylabel('Age') 
plt.title("Logistic Regression")
plt.show()
```
最后图表输出结果
![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/result1.png "plotresult")

可以看出时间对于死亡的影响非常的大，我觉得最后分类还是分的挺好的

