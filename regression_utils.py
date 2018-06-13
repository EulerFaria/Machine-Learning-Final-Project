import numpy as np
import pandas as pd
import random as rn
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse 


from keras.models import Sequential
from keras.layers.core import Activation,Dense,Dropout





def split_by_date(data,date_col,val_size):

    data.sort_values(by=date_col,inplace=True)
    
    data.reset_index(drop=True,inplace=True)
    
    split= 1- val_size
    tr,val = np.split(whole,[int(len(whole)*split)])    
    
    return tr,val 

def print_score(m,X_train,y_train,X_val,y_val):

    ypred = m.predict(X_train)
    ytrue = y_train
    print(f" Train Score:")
    print(f"RMSE: {rmse(ytrue,ypred):.6f}, MAE: {mae(ytrue,ypred):.6f}")
    if hasattr(m, 'oob_score_'): print(f"Oob score : {m.oob_score_:.6f}")    
        
        
    ypred = m.predict(X_val)
    ytrue = y_val
    print('-'*30)
    print(f" Val Score:")
    print(f"RMSE: {rmse(ytrue,ypred):.6f}, MAE: {mae(ytrue,ypred):.6f}")


    
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)



def print_report(m,X_train,y_train,X_val,y_val,X_test,y_test):

    ypred = m.predict(X_train)
    ytrue = y_train
    print(f" Train Score:")
    print(f"RMSE: {rmse(ytrue,ypred):.4f}, MAE: {mae(ytrue,ypred):.4f}")

    if hasattr(m, 'oob_score_'): print(f"Oob score : {m.oob_score_:.4f}")    
        
    ypred = m.predict(X_val)
    ytrue = y_val
    print('-'*30)
    print(f" Val Score:")
    print(f"RMSE: {rmse(ytrue,ypred):.4f}, MAE: {mae(ytrue,ypred):.4f}")

    ypred = m.predict(X_test)
    ytrue = y_test
    print('-'*30)
    print(f" Test Score:")
    print(f"RMSE: {rmse(ytrue,ypred):.4f}, MAE: {mae(ytrue,ypred):.4f}")

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False) 



# Taken from https://github.com/fastai/fastai
def add_datepart(df, fldname, drop=True, time=False):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace.

    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    time: If true time features: Hour, Minute, Second will be added.

    Examples:
    ---------

    >>> df = pd.DataFrame({ 'A' : pd.to_datetime(['3/11/2000', '3/12/2000', '3/13/2000'], infer_datetime_format=False) })
    >>> df

        A
    0   2000-03-11
    1   2000-03-12
    2   2000-03-13

    >>> add_datepart(df, 'A')
    >>> df

        AYear AMonth AWeek ADay ADayofweek ADayofyear AIs_month_end AIs_month_start AIs_quarter_end AIs_quarter_start AIs_year_end AIs_year_start AElapsed
    0   2000  3      10    11   5          71         False         False           False           False             False        False          952732800
    1   2000  3      10    12   6          72         False         False           False           False             False        False          952819200
    2   2000  3      11    13   0          73         False         False           False           False             False        False          952905600
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)



    


def MLP_Regressor(n_estimators,dp=None,input_dim=3):
    """
    Function to build the model.  
        n_estimators: List with the number of neurons for each layer
        dp: List with dropout values for each layer
        input_dim: Dimension of input ( features )
        
    returns: Keras sequential model instance;  
    """

    model = Sequential()
    model.add(Dense(n_estimators[0],activation='relu',
                    input_dim = input_dim,))
    
    if dp !=None:
        for i in range(1,len(n_estimators)):
            model.add(Dense(n_estimators[i],activation='relu',))
            model.add(Dropout(dp[i]))
    else:
        for i in range(1,len(n_estimators)):
            model.add(Dense(n_estimators[i],activation='relu',))
    
    
    model.add(Dense(1,activation='relu'))            
    
    return model



def rmse(ytrue,ypred):
    return np.sqrt(mse(ytrue,ypred))


 
    
def cross_valid(model,x,folds,metric,verbose=True):
    """ 
    This function does cross validation for general classifiers. 
        model: Sklearn model or customized model with fit and predict methods;
        x : Data as a numpy matrix containg with ***the last column as target***;
        folds: Number of folds;
        metrics : 'mae': mse,'rmse',
        verbose: Flag to print report over iterations;
        
    returns: List with scores over the folders
    """    

    score=[]
    

    kf = KFold(folds,shuffle=False,random_state=0) 


    i=0
    for train_index, test_index in kf.split(x):

        xtrain = x[train_index,:]
        xtest = x[test_index,:]

        model.fit(xtrain[:,:-1],xtrain[:,-1])

        ypred = model.predict(xtest[:,:-1])


        if metric == 'mae':
            score.append(mae(xtest[:,-1],ypred))
        elif metric == 'mse':
            score.append(mse(xtest[:,-1],ypred))
        else:
            score.append(rmse(xtest[:,-1],ypred))

        if verbose:
            print('-'*30)
            print(f'\nFold {i+1} out of {folds}')
            print(f'{metric}: {score[i]}')

        i+=1

    if verbose:
        print(f'\n Overall Score:')
        print(f'{metric}:    Mean: {np.mean(score)}   Std: {np.std(score)}')


    return score
        

        
# Adpated from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/        
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
    n_vars = 1 if type(data) is list else data.shape[1]
    var_n= data.columns.tolist()
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))                    
        names += [(var_n[j]+'(t-%d)' % ( i)) for j in range(n_vars)]


    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(var_n[j]+'(t)') for j in range(n_vars)]
        else:
            names += [(var_n[j]+'(t+%d)' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg



def dataTimeSeries(timesteps,df,predictors,target,dropnan,out=2,dropVars=True):
    """ 
    This function transforms a dataframe in a timeseries for surpervised learning.
        timesteps: Number of delays (i.e: timesteps =2 (t),(t-1),(t-2));
        df: Dataframe;
        predictors: List of columns in dataframe as features for the ML algorithm;
        target: Target of the supervised learning;
        dropnan: Flag to drop the NaN values after transforming the 
        out: Number of steps to forecast (i.e: out = 2 (t),(t+1));
        dropVars= Leave only the Target of the last timestep on the resulting dataframe;
    """    
    
    series = series_to_supervised(df[predictors+ [target]].copy(),timesteps,out,dropnan=dropnan)
    if dropnan==False:
        series.replace(pd.np.nan,0,inplace=True)
    
    # Dropping other variables:
    if dropVars:
        index = list(np.arange(series.shape[1]-2,
                               series.shape[1]-len(predictors)-2,
                               -1))

        labels = [item  for idx,item in enumerate(series.columns) 
                  if idx in index]

    #    print("Eliminando variáveis: {}".format(labels))
        series.drop(labels,axis=1,inplace=True)  

    return series
    
