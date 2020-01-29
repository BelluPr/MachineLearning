#Import the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing the dataset
data=pd.read_csv("Data.csv")
X=data.iloc[:,:-1].values
Y=data.iloc[:,3].values


#taking care of missing data

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


#encoding the catergircal data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Lable_X=LabelEncoder()
X[:,0]=Lable_X.fit_transform(X[:,0])



one_hot=OneHotEncoder(categorical_features=[0])
X=one_hot.fit_transform(X).toarray()
# splitting data into test and train

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

Y_L=LabelEncoder()
Y=Y_L.fit_transform(Y)



#Splitting the data into test and train
#ctrl+i for editor object

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)

#feature sacling

from sklearn.preprocessing import StandardScaler
SC_X=StandardScaler()
X_train=SC_X.fit_transform(X_train)
X_test=SC_X.transform(X_test)
