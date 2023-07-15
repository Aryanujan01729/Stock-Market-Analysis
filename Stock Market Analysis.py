#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential
from keras.layers import LSTM , concatenate,Dropout
from keras.layers import Dense
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras import regularizers
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_regression
from keras.regularizers import l1
from keras.regularizers import L1L2


# In[34]:

# In[5]:





# In[6]:


# def evaluate():
#     # Input the csv file
#     """
#     Sample evaluation function
#     Don't modify this function
#     """
#     df = pd.read_csv('sample_input.csv')
     
#     actual_close = np.loadtxt('sample_close.txt')
    
#     pred_close = predict_func(df)
    
#     # Calculation of squared_error
#     actual_close = np.array(actual_close)
#     pred_close = np.array(pred_close)
#     mean_square_error = np.mean(np.square(actual_close-pred_close))


#     pred_prev = [df['Close'].iloc[-1]]
#     pred_prev.append(pred_close[0])
#     pred_curr = pred_close
    
#     actual_prev = [df['Close'].iloc[-1]]
#     actual_prev.append(actual_close[0])
#     actual_curr = actual_close

#     # Calculation of directional_accuracy
#     pred_dir = np.array(pred_curr)-np.array(pred_prev)
#     actual_dir = np.array(actual_curr)-np.array(actual_prev)
#     dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

#     print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    
def evaluate():
    # Input the csv file
    df = pd.read_csv('NSEI.csv')
    inp_len = 50
    out_len = 2
    tot_len = inp_len+out_len
    
    sq_error = []
    direction = []
    for start in range(0, df.shape[0]-tot_len, inp_len+out_len):
        df1 = df.iloc[start:start+inp_len]
        df1 = df1.reset_index(drop=True)
        print(df1, file=open('output.txt', 'a'))
        
        actual_close = df['Close'].iloc[start+inp_len : start+tot_len]
        pred_close = predict_func(df1)
        
        #Calculation of squared error
        actual_close = np.array(actual_close)
        pred_close = np.array(pred_close)
        sq_error.extend(np.square(actual_close-pred_close))

        pred_prev = [df1['Close'].iloc[-1]]
        pred_prev.append(pred_close[0])
        pred_curr = pred_close

        actual_prev = [df1['Close'].iloc[-1]]
        actual_prev.append(actual_close[0])
        actual_curr = actual_close
        
        #Calculation of directional accuracy
        pred_dir = np.array(pred_curr)-np.array(pred_prev)
        actual_dir = np.array(actual_curr)-np.array(actual_prev)
        direction.append((pred_dir*actual_dir)>0)
        print(1)

    mean_square_error = np.mean(sq_error)
    dir_accuracy = np.mean(direction)*100    
    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')

# In[39]:


def predict_func(data):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    
    
    df=pd.read_csv('Stock_data.csv')
    df=df.interpolate()
    #df=df.dropna()
    #print(df.loc[20:30])
    # Taking One addition parameter of simple moving average 
    SMA=[]
    for i in range(len(df['Close'])):
        mean=np.mean(df['Close'].loc[i:i+20])
        SMA.append(mean)
    df['SMA(20)']=SMA
    #Dropping Date 
    df=df.drop(df.columns[0],axis=1)

    # ScalerStandard normalize the data
    scaler = StandardScaler()
    columns=df.columns
    norm_df1=pd.DataFrame(index=df.index)
    norm_df1[columns]=scaler.fit_transform(df[columns])

    arr_norm=np.array(norm_df1)

    arr = (norm_df1['Close'].shift(-1)).fillna(0)

    # Generate a sample dataset
    X,y=arr_norm,arr

    # Instantiate the LASSO model
    lasso = Lasso(alpha=0.1)  # You can adjust the regularization parameter alpha

    # Fit the model to the data
    lasso.fit(X, y)

    # Retrieve the coefficients
    coefficients = lasso.coef_

    print("Coefficients:", coefficients)
    #Making new Dataset with relevant Parameters 
    norm_df2=pd.DataFrame(index=df.index)
    norm_df2['Open']=norm_df1['Open']
    norm_df2['High']=norm_df1['High']
    norm_df2['Low']=norm_df1['Low']
    norm_df2['Close']=norm_df1['Close']
    norm_df2['SMA(20)']=norm_df1['SMA(20)']

    time_step=10
    train_X,train_Y=process_data(norm_df2,norm_df2['Close'],time_step)
    # Using multivariate LSTM
    model=Sequential()
    model.add(LSTM(50,return_sequences=True, input_shape=(11,5)))
    model.add(LSTM(25))
    Dropout(0.2)
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X,train_Y, epochs=10, batch_size=11)
    df2=data
    df2=df2.interpolate()
    df2=df2.drop(df2.columns[0],axis=1)
    df2=df2.drop(df2.columns[4],axis=1)
    #Finding Simple Moving Average of test Sample
    SMA1=[]
    for i in range(len(df2['Close'])):
        mean=np.mean(df2['Close'].loc[i:i+20])
        SMA1.append(mean)
    df2['SMA(20)']=SMA1
    df2=df2.drop(df2.columns[4],axis=1)
    #Normalizing Sample Input 
    sc = StandardScaler()
    df3=pd.DataFrame(index=df2.index)
    df3[df2.columns]=sc.fit_transform(df2[df2.columns])
    #Processing the data
    test_X,test_Y=process_data(df3,df3['Close'],time_step)
    pred_Y=model.predict(test_X)
    model.evaluate(test_X, test_Y)
    #finding mean and standard deviation of sample Input
    means=df2['Close'].mean()
    stdd=df2['Close'].std()
    pred_Y=unnormalize(pred_Y,means,stdd)
    test_Y=unnormalize(test_Y,means,stdd)
    mse =(mean_squared_error(test_Y, pred_Y))
    arr=test_X[-1:]
    #First Forecast Value
    fore_cast1=unnormalize(model.predict(arr),means,stdd)
    test_Y.append(fore_cast1)

    df3=pd.read_csv('Stock_data.csv')
    df3=df3.interpolate()
    df3['Close']=normalize(df3['Close'])

    train_X2,train_Y2=process_data(df3['Close'],df3['Close'],time_step)
    #Traing the model for second value 
    model1=Sequential()
    model1.add(LSTM(100,return_sequences=True, input_shape=(11,1)))
    model1.add(LSTM(50))
    model1.add(Dense(1))
    model1.compile(optimizer='adam', loss='mean_squared_error')
    model1.fit(train_X2,train_Y2, epochs=5, batch_size=11)

    test_Y=normalize(np.array(test_Y))
    t=np.array(test_Y)
    reshape=t.reshape(40,1)
    d=pd.DataFrame(reshape)
    
    test_X2,test_Y2=process_data(d,d,time_step)

    pred_Y2=model1.predict(train_X2)
    
    #model1.evaluate(test_X2)
    pred_Y2=unnormalize(pred_Y2,means,stdd)
    test_Y2=unnormalize(test_Y2,means,stdd)
    #Second Forecast value
    fore_cast2=unnormalize(model1.predict(test_X2[-1:]),means,stdd)
    return [fore_cast1,fore_cast2]


# In[40]:



# In[ ]:


if __name__== "__main__":
    evaluate()


# In[35]:


def normalize(p):
    arr=[]
    means=p.mean()
    stdd=p.std()
    for i in range(len(p)):
        unnorm = (p[i]- means)/stdd
        arr.append(unnorm)
        unnorm=0
    return arr


# In[36]:


def unnormalize(p,mean,std):
    arr=[]
    for i in range(len(p)):
        unnorm = (p[i]* std) + mean
        arr.append(unnorm)
        unnorm=0
    return arr


# In[37]:


def process_data(dataset,Y,time_step):
    data_X=[]
    data_Y=[]
    for i in range(len(dataset)-time_step-1):
        a=dataset.loc[i:(i+time_step)]
        b=Y.loc[i+time_step+1]
        data_X.append(a)
        data_Y.append(b)
    return np.array(data_X),np.array(data_Y)


# In[ ]:


def evaluate():
    # Input the csv file
    df = pd.read_csv('NSEI.csv')
    inp_len = 50
    out_len = 2
    tot_len = inp_len+out_len
    
    sq_error = []
    direction = []
    for start in range(0, df.shape[0]-tot_len, inp_len+out_len):
        df1 = df.iloc[start:start+inp_len]
        df1 = df1.reset_index(drop=True)
        print(df1, file=open('output.txt', 'a'))
        
        actual_close = df['Close'].iloc[start+inp_len : start+tot_len]
        pred_close = predict_func(df1)
        
        #Calculation of squared error
        actual_close = np.array(actual_close)
        pred_close = np.array(pred_close)
        sq_error.extend(np.square(actual_close-pred_close))

        pred_prev = [df1['Close'].iloc[-1]]
        pred_prev.append(pred_close[0])
        pred_curr = pred_close

        actual_prev = [df1['Close'].iloc[-1]]
        actual_prev.append(actual_close[0])
        actual_curr = actual_close
        
        #Calculation of directional accuracy
        pred_dir = np.array(pred_curr)-np.array(pred_prev)
        actual_dir = np.array(actual_curr)-np.array(actual_prev)
        direction.append((pred_dir*actual_dir)>0)
        print(1)

    mean_square_error = np.mean(sq_error)
    dir_accuracy = np.mean(direction)*100    
    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')

