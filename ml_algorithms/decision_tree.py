# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as ssn
import random
from sklearn.model_selection import train_test_split


#%%
df = pd.read_csv("./data/Iris.csv")
df = df.drop("Id", axis = 1)
df = df.rename(columns = {"Species":"Label"})
df.head()


#%%
x = df.drop("Label", axis = 1).values.tolist()
y = df["Label"].values.tolist()
train_x, train_y, test_x, test_y = train_test_split(x,y,test_size = 0.3)


#%%
if __name__ == "__main__":
    train, test = train_test_split()
    tree = DecisionTree()
    accuracy = CalculateAccuray()

