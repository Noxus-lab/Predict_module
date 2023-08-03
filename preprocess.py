from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import feature as f
def subOutlier (df : pd.DataFrame,train_data_num, substitute=np.nan) :
    df_out=df.copy()
    
    clf = IsolationForest(random_state=0).fit(df_out[:train_data_num])
    outliers1 = clf.predict(df_out[:train_data_num])
    
    outliers2 = clf.predict(df_out[train_data_num:])
    outliers =  np.concatenate([outliers1, outliers2])
    for col in df_out.columns:
        # Replace outliers with the median of the respective feature
        df_out.loc[outliers == -1, col] = substitute

    return df_out

def fillNan(df:pd.DataFrame, train_data_num, neighbors=3): 
    df_out = df.copy()
    imputer = KNNImputer(n_neighbors=neighbors) 
    imputed = imputer.fit_transform(df_out[:train_data_num])
    df_out[:train_data_num] = pd.DataFrame(imputed, columns=df.columns)
    # imputer = KNNImputer(n_neighbors=neighbors) 
    imputed = imputer.fit_transform(df_out[train_data_num:])
    df_out[train_data_num:] = pd.DataFrame(imputed, columns=df.columns)
    return df_out

def standardScaler(df: pd.DataFrame, scale_list, ratio):
    df_out = df.copy()
    
    scaler = StandardScaler()
    # Fit the scaler to the DataFrame
    
    for col in scale_list:
        train_id = int(len(df_out[col]) * ratio)
        scaler.fit(df_out[col][:train_id].to_numpy().reshape(-1,1))

        # Transform the DataFrame to standardize the numerical features
        standardized_data = scaler.transform(df_out[col].to_numpy().reshape(-1,1))
        df_out[col]=standardized_data
    

    return df_out

if __name__ == "__main__":
    # ------------------read data usage-----------------
    stock_list=f.read_csv("./tech_1.csv","./data/")
    # print(stock_list['AAPL'].head())
    #-------------------sub outlier---------------------
    df= subOutlier(stock_list['AAPL'],500)
    # print(df)
    
    #-------------------fillNaN-------------------------
    df = fillNan(df,500)
    # print(df)

    #-------------------standardScaler------------------
    df = standardScaler(stock_list['AAPL'],['Open','Close'],0.8)
    print(df.head())