from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
import feature as f
def subOutlier (df : pd.DataFrame,train_data_num, substitute=np.nan) :
    """
    Substitute outliers in the DataFrame using Isolation Forest.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        train_data_num (int): The number of rows in the training data.
        substitute: The value to substitute outliers with. Default is np.nan.

    Returns:
        pd.DataFrame: A new DataFrame with outliers substituted.
    """
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
    """
    Fill missing values in the DataFrame using KNNImputer.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        train_data_num (int): The number of rows in the training data.
        neighbors (int, optional): The number of neighbors to consider. Default is 3.

    Returns:
        pd.DataFrame: A new DataFrame with missing values filled.
    """
    df_out = df.copy()
    imputer = KNNImputer(n_neighbors=neighbors) 
    imputed = imputer.fit_transform(df_out[:train_data_num])
    df_out[:train_data_num] = pd.DataFrame(imputed, columns=df.columns)
    # imputer = KNNImputer(n_neighbors=neighbors) 
    imputed = imputer.fit_transform(df_out[train_data_num:])
    df_out[train_data_num:] = pd.DataFrame(imputed, columns=df.columns)
    return df_out

def standardScaler(df: pd.DataFrame, scale_list, ratio):
    """
    Standardize numerical features in the DataFrame using StandardScaler.

    Parameters:
        df (pd.DataFrame): The DataFrame containing data.
        scale_list (list): List of column names to be scaled.
        ratio (float): The ratio of training data for fitting the scaler.

    Returns:
        pd.DataFrame: A new DataFrame with standardized numerical features.
    """
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
    print(stock_list['AAPL'].head())
    # -------------------sub outlier---------------------
    df= subOutlier(stock_list['AAPL'],500)
    print(df)
    
    # -------------------fillNaN-------------------------
    df = fillNan(df,500)
    print(df)

    # -------------------standardScaler------------------
    df = standardScaler(stock_list['AAPL'],['Open','Close'],0.8)
    print(df.head())