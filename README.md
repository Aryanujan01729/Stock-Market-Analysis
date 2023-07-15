
# Project Title

A brief description of what this project does and who it's for


## Python Libraries and Packages 
numpy ,pandas,matplotlib,tensorflow,keras,Sequential,LSTM , concatenate,Dropout,Dense,Lasso,StandardScaler,regularizers,make_regression
## Documentation

[Documentation](https://linktodocumentation)

I have used various libraries and packages to carry out the task.
From the given data i first found out the Simple Moving Average (SMA) which i found out is important parameter for stock prediction.
Then i performed LASSO regression for parameter regression . LASSO regression results told that there is no contribution of Volume and Adj close so we removed those parameters and taken Open ,Close,High,Low ,SMA 
then we Normalized the data using ScalerStandard, and processed the data to create moving window 
then we constructed the multivariate LSTM model and train our data 
this model forecast one value but we are asked to forecast two values so we constructed another LSTM model that is univariate and trained on close parameter only 
we forecast other value by this way.