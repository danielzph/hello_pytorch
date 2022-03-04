import pandas as pd
import numpy as np



df=pd.read_csv("corr_data01.csv")

data_f=df[df.columns[0:5]]
train=data_f.values
newtrain=[]
for i in range(len(train)):
    if (i+1 )%30==0:
        for j in range(i-29,i-7):
            newtrain.append(train[j])


df=pd.DataFrame(newtrain)
df.to_csv("make_csv.csv",header=True,index=False,encoding="utf-8")
