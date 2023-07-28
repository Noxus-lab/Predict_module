import numpy as np
import pandas as pd
import os

def read_csv(symbol_path, stock_path='./'):
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
            stock_list[symbols[i]]=df
    return stock_list
def sub_columns(df1,df2,col):
    ans=[]
    ans_df=pd.DataFrame()
    for i in range(len(df1[col])):
        if(df1[col][i]!=np.nan and df2[col][i]!=np.nan):
            ans.append(df1[col][i]-df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col]=ans
    return ans_df
def add_columns(df1,df2,col):
    ans=[]
    ans_df=pd.DataFrame()
    for i in range(len(df1[col])):
        if(df1[col][i]!=np.nan and df2[col][i]!=np.nan):
            ans.append(df1[col][i]+df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col]=ans
    return ans_df
def multiply_columns(df1,df2,col):
    ans=[]
    ans_df=pd.DataFrame()
    for i in range(len(df1[col])):
        if(df1[col][i]!=np.nan and df2[col][i]!=np.nan):
            ans.append(df1[col][i]*df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col]=ans
    return ans_df

def divide_columns(df1,df2,col):
    ans=[]
    ans_df=pd.DataFrame()
    for i in range(len(df1[col])):
        if(df1[col][i]!=np.nan and df2[col][i]!=np.nan):
            ans.append(df1[col][i]/df2[col][i])
        else:
            ans.append(np.nan)
    ans_df[col]=ans
    return ans_df
#Time series function
class ts_func:
    def __init__(self, df):
        self.df = df.copy()
        self.date_copy = df.index
    def update(self, df):
        self.df = df.copy()
        self.date_copy = df.index
    def ts_max(self,col,window):
        ans=[]
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].max())
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_min(self,col,window):
        ans=[]
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].min())
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_rank(self,col,window,constant=0):
        ans=[]
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].rank()[-1]+constant)
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    

    def ts_corr(self,col1,df2,col2,window):
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].corr(df2[col2][i-window:i]))
        ans_df=pd.DataFrame()
        ans_df[col1]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_cov(self,col1,df2,col2,window):
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].cov(df2[col2][i-window:i]))
        ans_df=pd.DataFrame()
        ans_df[col1]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_var(self,col1,window):
        ans = []
        for i in range(len(self.df[col1])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col1][i-window:i].var())
        ans_df=pd.DataFrame()
        ans_df[col1]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_std_dev(self,col,window):
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].std())
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_zscore(self, col, window):
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append((self.df[col][i]-self.df[col][i-window:i].mean())/self.df[col][i-window:i].std())
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def ts_mean(self, col, window):
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window:i].mean())
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_delay(self, col, window):
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i-window])
        ans_df=pd.DataFrame()
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    
    def ts_delta(self,col,window):
        ans_df=pd.DataFrame()
        ans = []
        for i in range(len(self.df[col])):
            if(i<window):
                ans.append(np.nan)
            else:
                ans.append(self.df[col][i]-self.df[col][i-window])
        ans_df[col]=ans
        ans_df['Date']=self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
    def pct_change(self,col):
        ans_df=pd.DataFrame()
        ans_df[col+'_growth_rate'] = self.df[col].pct_change()
        ans_df['Date'] = self.date_copy
        ans_df.set_index('Date', inplace=True)
        return ans_df
        
#Cross Sectional Operator
class cs:
    def __init__(self,df_list):
        self.df_list=df_list
        for name in df_list:
            self.date_copy=df_list[name].index
            break
        # print(self.date_copy)
    def rank(self,col):
        record_df = pd.DataFrame()
        record_df['Date']=self.date_copy
        record_df.set_index('Date', inplace=True)
        count=0
        for df in self.df_list:
            record_df[str(count)]=self.df_list[df][col].copy()
            count+=1
        for i in range(record_df.shape[0]):
            record_df.T.iloc[:,i]=record_df.T.iloc[:,i].rank()
        final_df_list=[]
        for i in range(len(self.df_list)):
            ans_df=pd.DataFrame()
            ans_df[col]=record_df.iloc[:,i]
            final_df_list.append(ans_df)
        return final_df_list
    def normalize(self, col, useStd=False):
        record_df = pd.DataFrame()
        record_df['Date']=self.date_copy
        record_df.set_index('Date', inplace=True)
        count=0
        for df in self.df_list:
            record_df[str(count)]=self.df_list[df][col]
            count+=1
        
        for i in range(record_df.shape[0]):
            if(useStd==False):
                record_df.T.iloc[:,i]=(record_df.T.iloc[:,i]-record_df.T.iloc[:,i].mean())/record_df.T.iloc[:,i].std()
            else:
                record_df.T.iloc[:,i]=(record_df.T.iloc[:,i]-record_df.T.iloc[:,i].mean())
        final_df_list=[]
        for i in range(len(self.df_list)):
            ans_df=pd.DataFrame()
            ans_df[col]=record_df.iloc[:,i]
            final_df_list.append(ans_df)
        return final_df_list
def create_feature_list_ts(name_list, alpha_list):
    df=pd.DataFrame()
    for idx,name in enumerate(name_list):
        df[name]=alpha_list[idx].iloc[:,0]
    return df
if __name__ == "__main__":
    # ------------------read data usage-----------------
    stock_list=read_csv("./tech_1.csv","./data/")
    # ------------------cross sectional usage-----------------
    CS=cs(stock_list)
    # final_df_list=CS.rank('Close')
    # final_df_list=CS.normalize('Close')
    
    # ------------------time series usage-----------------
    tmp=ts_func(stock_list['AAPL'])
    # print(tmp.ts_cov('Open',stock_list['GOOG'],'Open',5))
    # print(tmp.ts_var('Open',5))
    # print(tmp.pct_change('Open'))
    # print(tmp.ts_max('Open',5))
    # print(tmp.ts_min('Open',5))
    # print(tmp.ts_rank('Open',5))
    # print(tmp.ts_corr('Open',stock_list['GOOG'],'Open',5))
    # print(tmp.ts_std_dev('Open',5))
    # print(tmp.ts_zscore('Open',5))
    # print(tmp.ts_mean('Volume',5))
    # print(tmp.ts_delay('Open',5))
    # print(tmp.ts_delta('Open',5))
    # ------------------final usage-----------------
    # df=create_feature_list_ts(['a','b'],[tmp.ts_max('Open',5),tmp.ts_min('Open',5)])
    # print(df)
