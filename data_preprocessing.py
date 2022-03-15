import numpy as np
import pandas as pd
from scipy import stats
import math
"""
Load data from MaternalDataset.csv and preprocess them

"""


#Load the data in pandas dataframe
df = pd.read_csv("./MaternalDataset.csv")
attributeNames = ["Age", "Systolic BP", "Diastolic BP", "Blood Glucose", 
                  "Body Temperature", "Heart Rate", "Risk Level"]
raw_data = df.values
classes = raw_data[:-2,-1]
classes = list(set(classes))

classNames = {"low risk":1,"mid risk":2, "high risk":3}
y = np.asarray([classNames[value] for value in raw_data[:,-1]])

N,M = raw_data.shape
data = np.asarray(raw_data,)


#Transform the values of the last attribute to numerical
for i in range(N):
    if data[i,6]=="low risk":
        data[i,6]=1
    elif data[i,6]=="mid risk":
        data[i,6]=2
    else:
        data[i,6]=3             

#Checking for missing values
count = 0
for i in range(N):
    for j in range(M):
        if not (isinstance(data[i,j],int) or isinstance(data[i,j],float)):
            count = count + 1


#Tr
t_data = data.T
K,L=t_data.shape
#Create an array for the variance of the attributes 
var = []
for i in range(K):
    var.append(round(t_data[i].var(),2))
#print("The array of the variance is: " , var)
#Create an array for the svd of the attributes
std = []
for i in range(K):
    std.append(round(math.sqrt(var[i]),2))
#print("The array of the standard deviation is: " , std)
#Substract the mean and divide with standard deviation
X = raw_data - np.ones((N,1))*raw_data.mean(axis=0)
for i in range(M):
    X[:i] = X[:i]*(1/std[i])
#print(X)
#X = X * (1 / np.std(X, 0))
# Uncomment
#print("The number of missing of corrupted values is %i." %count)
#print(data)


