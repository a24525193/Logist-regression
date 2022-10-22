# Logist-regression
Pattern Recognition Assignment 1 using Python


## Data Set

https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

This dataset contains the medical records of 299 patients who had heart failure, collected during their follow-up period, where each patient profile has 13 clinical features.

Attribute Information:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)



## Package used

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```



## Pre-analysis

The data is imported as dataframe, and there is no missing value.

Check the correlation coefficient of the target value with the heat map, then select the features.

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/heatmap.png "heatmap")

Finally, I choose 'TIME' with the highest correlation coefficient with DEATH_EVENT and 'AGE' with better resolution.

Logistic regression classification will be made based on these two characteristics to predict whether the death event occurs




## Training and Prediction

Set 'TIME' and 'AGE' to X,
'DEATH_ EVENT' is set to y, and then X and y are divided into training set and test set at a ratio of 8:2.

And use sklearn's logistic regression package to train the model.

Finally, the prediction results show that precision and F1 score reaches 0.88.



## Create Plot


Create scatter chart and put in the two features.



> Decision boundary

1. Create a function, set the maximum and minimum values, and attach the edge filling.

1. Use prediction function to predict

1. Draw the decision boundary in the plot

Mark the display icon to 'o' and 'x', and also change the color to green and blue. In order to separately the classification for more obvious distinction.


## Final output results

![GITHUB](https://github.com/a24525193/Logist-regression/blob/main/result1.png "plotresult")

可以看出时间对于死亡的影响非常的大，我觉得最后分类还是分的挺好的
