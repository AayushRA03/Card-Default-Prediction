import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC #support vector machine classifier
from sklearn.neural_network import MLPClassifier #multi layer perceptron classifier or neural network

data = pd.read_csv("C:\\Users\\Aayush Rathore\\Downloads\\archive\\UCI_Credit_Card.csv")
print(data)
pd.set_option('display.max_columns',None)
print("-----------------DATA-----------------")
print(data)#data is clean and ready
print("-----------------IFORMATION-----------------")
print(data.info())

# one hot encoding
def one_hot_encoding(df,column_dict):#this  dict is for the pair for collumn name and prefix 
    df = df.copy()
    for column,prefix in column_dict.items():  
       dummies = pd.get_dummies(df[column],prefix=prefix)
       df = pd.concat([df,dummies],axis=1)#concatenate the dummies with the original dataframe
       df = df.drop(column,axis=1)#drop the original column
    return df

#preprocessing
def preprocess_input(df):
    #so  that we dont lose original data
    df = df.copy()

    #dropping id column
    df = df.drop('ID',axis=1)
    df=one_hot_encoding(
    df,{
        'EDUCATION':'EDU',
        'MARRIAGE':'MAR',  
    }
    )
    #split df into x and y
    y = df['default.payment.next.month'].copy()
    X = df.drop('default.payment.next.month',axis=1).copy()

    #scaling(bringing to same range/standarizing the columns)
    #scale x with standard scalar
    scaler = StandardScaler()#apply transdormation so that each collumn has mean 0 and unit variance 
    X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns)#returns numpy array

    return X,y

X,y = preprocess_input(data)
print("-----------------X-----------------")
print(X)
print("-----------------X.mean()-----------------")
print(X.mean())
print("-----------------y-----------------")
print(y)

#dict for mapping collumns name to prefix
#{'EDUCATION':'EDU'}.items()
#a=pd.get_dummies(X['EDUCATION'],prefix='EDU')
#print(a)#0 to 6 unique value of education

##visualization

corr = data.corr()#calculate correlation matrix
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True,vmin=-1.0,cmap='mako')
plt.title('Correlation heatmap')
plt.show()
unique_values = {column:len(X[column].unique())for column in X.columns}
print("-----------------unique_values-----------------")
print(unique_values)

#training
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.7,random_state=42)#train size is data to include in training and random state for shuffling the data
models={
    LogisticRegression():"   Logistic Regression",
    SVC():               "Support vector machine",
    MLPClassifier():     "        Neural Network",
}
for model in models.keys(): 
    model.fit(X_train,y_train)
    print(f"{models[model]} trained")
    print("-----------------score-----------------")

    for mode,name in models.items():
        print(name + ":{:.4f}%".format(model.score(X_test, y_test)*100))

