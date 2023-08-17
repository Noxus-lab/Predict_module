import numpy as np
import pandas as pd
import os

def read_csv(symbol_path, stock_path='./'):
    """
    Reads stock data CSV files and returns a dictionary of DataFrames.

    Parameters:
        symbol_path (str): Path to the CSV file containing stock symbols.
        stock_path (str, optional): Path to the directory containing stock data CSV files. Defaults to './'.

    Returns:
        dict: A dictionary where keys are stock symbols and values are DataFrames containing stock data.
    """
    symbols = pd.read_csv(symbol_path)
    symbols = symbols['Symbol']
    num_arrays = len(symbols)
    stock_list ={}
    for i in range(num_arrays):
        if os.path.exists(stock_path+symbols[i]+".csv"):
            df = pd.read_csv(stock_path+symbols[i]+".csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.drop(columns=['Dividends', 'Stock Splits'],axis=1)
            stock_list[symbols[i]] = df
    return stock_list

def read_fund(symbol_path, stock_path='./'):
    """
    Reads stock fundamental data CSV files and returns a dictionary of DataFrames.

    Parameters:
        symbol_path (str): Path to the CSV file containing stock symbols.
        stock_path (str, optional): Path to the directory containing stock data CSV files. Defaults to './'.

    Returns:
        dict: A dictionary where keys are stock symbols and values are DataFrames containing stock fundamental data.
    """
    symbols = pd.read_csv(symbol_path)
    symbols = symbols['Symbol']
    num_arrays = len(symbols)
    stock_list ={}
    for i in range(num_arrays):
        if os.path.exists(stock_path+"fundamental_"+symbols[i]+".csv"):
            df = pd.read_csv(stock_path+"fundamental_"+symbols[i]+".csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            stock_list[symbols[i]]=df
    return stock_list

def sub_columns(df1,df2,col):
    """
    Subtract corresponding values of a specific column between two DataFrames and return the result.

    Parameters:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
        col (str): Column name for subtraction.

    Returns:
        pandas.DataFrame: Resulting DataFrame with subtracted values.
    """
    ans = []
    ans_df = pd.DataFrame()
    ans_df['Date'] = df1.index
    ans_df.set_index('Date', inplace=True)
    for i in range(len(df1[col])):
        if(df1[col][i]!=np.nan and df2[col][i]!=np.nan):
            ans.append(df1[col][i]-df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col] = ans
    return ans_df

def add_columns(df1,df2,col):
    """
    Add corresponding values of a specific column between two DataFrames and return the result.

    Parameters:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
        col (str): Column name for addition.

    Returns:
        pandas.DataFrame: Resulting DataFrame with added values.
    """
    ans = []
    ans_df = pd.DataFrame()
    ans_df['Date'] = df1.index
    ans_df.set_index('Date', inplace=True)
    for i in range(len(df1[col])):
        if(df1[col][i] != np.nan and df2[col][i] != np.nan):
            ans.append(df1[col][i]+df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col] = ans
    return ans_df

def multiply_columns(df1,df2,col):
    """
    Multiply corresponding values of a specific column between two DataFrames and return the result.

    Parameters:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
        col (str): Column name for multiplication.

    Returns:
        pandas.DataFrame: Resulting DataFrame with multiplied values.
    """
    ans = []
    ans_df = pd.DataFrame()
    ans_df['Date'] = df1.index
    ans_df.set_index('Date', inplace=True)
    for i in range(len(df1[col])):
        if(df1[col][i] != np.nan and df2[col][i] != np.nan):
            ans.append(df1[col][i]*df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col] = ans
    return ans_df

def divide_columns(df1,df2,col):
    """
    Divide corresponding values of a specific column between two DataFrames and return the result.

    Parameters:
        df1 (pandas.DataFrame): First DataFrame.
        df2 (pandas.DataFrame): Second DataFrame.
        col (str): Column name for division.

    Returns:
        pandas.DataFrame: Resulting DataFrame with divided values.
    """
    ans = []
    ans_df = pd.DataFrame()
    ans_df['Date'] = df1.index
    ans_df.set_index('Date', inplace = True)
    for i in range(len(df1[col])):
        if(df1[col][i] != np.nan and df2[col][i] != np.nan):
            ans.append(df1[col][i]/df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col]=ans
    return ans_df
class ts_func:
    """
    Time series function class for performing various time series operations on a DataFrame.
    """
    def __init__(self, df):
        """
        Initialize the ts_func class.

        Parameters:
            df (pandas.DataFrame): Input DataFrame containing time series data.
        """
        self.df = df.copy()
        self.date_copy = df.index
    def update(self, df):
        """
        Update the DataFrame to be used for time series operations.

        Parameters:
            df (pandas.DataFrame): New DataFrame for time series operations.
        """
        self.df = df.copy()
        self.date_copy = df.index
    def ts_max(self,col,window):
        """
        Calculate the rolling maximum of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating maximum.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling maximum values.
        """
        ans=[]
        for i in range(len(self.df[col])):
            if(i < window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].max())
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_min(self,col,window):
        """
        Calculate the rolling minimum of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating minimum.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling minimum values.
        """
        ans=[]
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].min())
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_rank(self,col,window,constant=0):
        """
        Calculate the rolling rank of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating rank.
            window (int): Size of the rolling window.
            constant (float, optional): Constant to be added to the rank. Default is 0.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling rank values.
        """
        ans=[]
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].rank()[-1] + constant)
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace = True)
        return ans_df
    

    def ts_corr(self,col1,df2,col2,window):
        """
        Calculate the rolling correlation between two columns from different DataFrames within a given window.

        Parameters:
            col1 (str): Column name from the first DataFrame.
            df2 (pandas.DataFrame): Second DataFrame.
            col2 (str): Column name from the second DataFrame.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling correlation values.
        """
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].corr(df2[col2][i-window:i]))
        ans_df = pd.DataFrame()
        ans_df[col1] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_cov(self,col1,df2,col2,window):
        """
        Calculate the rolling covariance between two columns from different DataFrames within a given window.

        Parameters:
            col1 (str): Column name from the first DataFrame.
            df2 (pandas.DataFrame): Second DataFrame.
            col2 (str): Column name from the second DataFrame.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling covariance values.
        """
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].cov(df2[col2][i-window:i]))
        ans_df = pd.DataFrame()
        ans_df[col1] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_var(self,col1,window):
        """
        Calculate the rolling variance of a specific column within a given window.

        Parameters:
            col1 (str): Column name for calculating variance.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling variance values.
        """
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].var())
        ans_df = pd.DataFrame()
        ans_df[col1] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_std_dev(self,col,window):
        """
        Calculate the rolling standard deviation of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating standard deviation.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling standard deviation values.
        """
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].std())
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_zscore(self, col, window):
        """
        Calculate the rolling z-score of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating z-score.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling z-score values.
        """
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append((self.df[col][i]-self.df[col][i-window:i].mean())/self.df[col][i-window:i].std())
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_mean(self, col, window):
        """
        Calculate the rolling mean of a specific column within a given window.

        Parameters:
            col (str): Column name for calculating mean.
            window (int): Size of the rolling window.

        Returns:
            pandas.DataFrame: DataFrame with calculated rolling mean values.
        """
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].mean())
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_delay(self, col, window):
        """
        Create a new column with delayed values of a specific column.

        Parameters:
            col (str): Column name for creating a delayed column.
            window (int): Number of time steps to delay.

        Returns:
            pandas.DataFrame: DataFrame with a new column containing delayed values.
        """
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window])
        ans_df = pd.DataFrame()
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_delta(self,col,window):
        """
        Calculate the difference between a column's value and its value 'window' time steps ago.

        Parameters:
            col (str): Column name for calculating delta.
            window (int): Number of time steps for the delta calculation.

        Returns:
            pandas.DataFrame: DataFrame with calculated delta values.
        """
        ans_df = pd.DataFrame()
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i]-self.df[col][i-window])
        ans_df[col] = ans
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def pct_change(self,col):
        """
        Calculate the percentage change of a specific column.

        Parameters:
            col (str): Column name for calculating percentage change.

        Returns:
            pandas.DataFrame: DataFrame with calculated percentage change values.
        """
        ans_df = pd.DataFrame()
        ans_df[col+'_growth_rate'] = self.df[col].pct_change()
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
        
#Cross Sectional Operator
class cs:
    """
    Cross Sectional Operator class for performing operations across multiple DataFrames.

    Parameters:
        df_list (dict): A dictionary of DataFrame objects.

    Attributes:
        df_list (dict): A dictionary of DataFrame objects.
        date_copy (pandas.Index): Copy of index from one of the DataFrame objects.
    """
    def __init__(self,df_list):
        """
        Initialize the cs object.

        Parameters:
            df_list (dict): A dictionary of DataFrame objects.
        """
        self.df_list = df_list
        for name in df_list:
            self.date_copy = df_list[name].index
            break
        # print(self.date_copy)
    def rank(self,col):
        """
        Rank the values in the specified column across all DataFrames.

        Parameters:
            col (str): Column name to be ranked.

        Returns:
            list of pandas.DataFrame: A list of DataFrames with ranked values for the specified column.
        """
        record_df = pd.DataFrame()
        record_df['Date'] = self.date_copy
        record_df.set_index('Date', inplace=True)
        count = 0
        for df in self.df_list:
            record_df[str(count)] = self.df_list[df][col].copy()
            count += 1
        for i in range(record_df.shape[0]):
            record_df.T.iloc[:,i]=record_df.T.iloc[:,i].rank()
        final_df_list = []
        for i in range(len(self.df_list)):
            ans_df = pd.DataFrame()
            ans_df[col] = record_df.iloc[:,i]
            final_df_list.append(ans_df)
        return final_df_list
    def normalize(self, col, useStd=False):
        """
        Normalize the values in the specified column across all DataFrames.

        Parameters:
            col (str): Column name to be normalized.
            useStd (bool, optional): If True, normalize using standard deviation, else normalize using mean and standard deviation.

        Returns:
            list of pandas.DataFrame: A list of DataFrames with normalized values for the specified column.
        """
        record_df = pd.DataFrame()
        record_df['Date'] = self.date_copy
        record_df.set_index('Date', inplace=True)
        count = 0
        for df in self.df_list:
            record_df[str(count)] = self.df_list[df][col]
            count += 1
        
        for i in range(record_df.shape[0]):
            if(useStd==False):
                record_df.T.iloc[:,i] = (record_df.T.iloc[:,i]-record_df.T.iloc[:,i].mean())/record_df.T.iloc[:,i].std()
            else:
                record_df.T.iloc[:,i] = (record_df.T.iloc[:,i]-record_df.T.iloc[:,i].mean())
        final_df_list = []
        for i in range(len(self.df_list)):
            ans_df = pd.DataFrame()
            ans_df[col] = record_df.iloc[:,i]
            final_df_list.append(ans_df)
        return final_df_list
def create_feature_list_ts(name_list, alpha_list):
    """
    Create a DataFrame from a list of alpha DataFrames.

    Parameters:
        name_list (list of str): List of names for the alpha DataFrames.
        alpha_list (list of pandas.DataFrame): List of alpha DataFrames.

    Returns:
        pandas.DataFrame: A DataFrame containing columns from the alpha DataFrames.
    """
    df = pd.DataFrame()
    for idx,name in enumerate(name_list):
        df[name] = alpha_list[idx].iloc[:,0]
    return df
if __name__ == "__main__":
    # ------------------read data usage-----------------
    stock_list = read_csv("./tech_1.csv","./data/")
    print(stock_list)
    stock_list = read_fund("./tech_1.csv","./data/")
    print(stock_list)
    # ------------------cross sectional usage-----------------
    CS = cs(stock_list)
    final_df_list=CS.rank('Close')
    final_df_list=CS.normalize('Close')
    
    # ------------------time series usage-----------------
    tmp = ts_func(stock_list['AAPL'])
    print(tmp.ts_cov('Open',stock_list['GOOG'],'Open',5))
    print(tmp.ts_var('Open',5))
    print(tmp.pct_change('Open'))
    print(tmp.ts_max('Open',5))
    print(tmp.ts_min('Open',5))
    print(tmp.ts_rank('Open',5))
    print(tmp.ts_corr('Open',stock_list['GOOG'],'Open',5))
    print(tmp.ts_std_dev('Open',5))
    print(tmp.ts_zscore('Open',5))
    print(tmp.ts_mean('Volume',5))
    print(tmp.ts_delay('Open',5))
    print(tmp.ts_delta('Open',5))
    # ------------------final usage-----------------
    df=create_feature_list_ts(['a','b'],[tmp.ts_max('Open',5),tmp.ts_min('Open',5)])
    print(df)
    
    
