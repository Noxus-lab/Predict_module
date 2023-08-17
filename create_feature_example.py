import feature as ft
import pandas as pd
import preprocess as p


'''
This file is an example of how we can use the feature.py and preprocess.py to create the corresponding csv files
'''
if __name__ == "__main__":
    train_data_num = 240 # train data number for one interval
    test_data_num = 10 # test data number for one interval
    data_dir = "./six_year_extra/" # where you put your basic stock data downloaded from yfinance
    screener_path = './screener_top_41_tmp.csv' 
    
    symbols = pd.read_csv(screener_path)
    symbols = symbols['Symbol']

    stock_list=ft.read_csv(screener_path, data_dir)
    total_data_num = len(list(stock_list.values())[0])# total data number
    for name in symbols:
        id=0
        L=len(str(int((total_data_num-60-train_data_num)/test_data_num))) # maximum length of the number 
        for i in range(0, total_data_num-60-train_data_num, test_data_num):
            id+=1
            name_list = ['Returns', 'ReturnVelocity', 'ReturnAcceleration', 'SmaDiffWeekMonth', 'VolumeVelocity', 'IsBiggestVolume', 'WeekMomentum', 'MonthMomentum'
                , 'SeasonVolatility']
            '''
            Create Returns dataframe
            '''
            stock_df = stock_list[name][i:i+train_data_num+test_data_num+60]
            obj = ft.ts_func(stock_df)
            

            returns = 100 * (stock_df['Close'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)
            returns_df = pd.DataFrame()
            returns_df['Returns'] = returns
            stock_df['Returns'] = returns
            obj.update(stock_df)
            
            '''
            Create Returns Velocity dataframe
            '''
            return_velocity = obj.ts_delta('Returns', 1)
            stock_df['ReturnVelocity'] = return_velocity
            obj.update(stock_df)

            '''
            Create Returns Acceleration dataframe
            '''
            return_acceleration = obj.ts_delta('ReturnVelocity', 1)

            '''
            Create SmaDiffWeekMonth dataframe
            '''
            sma_diff_week_month = obj.ts_mean('Close', 5) - obj.ts_mean('Close', 20)
            
            '''
            Create VolumeVelocity dataframe
            '''
            volume_velocity = 100 * (stock_df['Volume'] - stock_df['Volume'].shift(1)) / stock_df['Volume'].shift(1)
            volume_velocity_df = pd.DataFrame()
            volume_velocity_df['VolumeVelocity'] = volume_velocity

            '''
            Create IsBiggestVolume dataframe
            '''
            tmp_df = obj.ts_rank('Volume', 5)
            tmp_df['Volume'] = tmp_df['Volume'].apply(lambda x: 1 if x == 1 else 0)
            is_biggest_volume = tmp_df

            '''
            Create WeekMomentum dataframe
            '''
            week_momentum = ft.divide_columns(obj.ts_delta('Close', 5), obj.ts_delay('Close', 5), 'Close')
            
            '''
            Create MonthMomentum dataframe
            '''
            month_momentum = ft.divide_columns(obj.ts_delta('Close', 20), obj.ts_delay('Close', 20), 'Close')

            '''
            Create SeasonVolatility dataframe
            '''
            season_volatility = obj.ts_var('Returns', 60)

            '''
            Create the dataframe containing the features we devise
            '''
            alpha_list = [returns_df, return_velocity, return_acceleration, sma_diff_week_month, volume_velocity_df, is_biggest_volume, week_momentum, month_momentum
                        , season_volatility]

            df = ft.create_feature_list_ts(name_list,alpha_list)
            df = df.dropna()
            
            '''
            Standard scale the columns we specify 
            '''
            scale_list = ['SmaDiffWeekMonth', 'VolumeVelocity', 'WeekMomentum', 'MonthMomentum', 'SeasonVolatility']
            
            df = p.standardScaler(df, scale_list, train_data_num/(train_data_num+test_data_num))
            
            '''
            Handle the name of the output csv file and output(this is just a example)
            '''
            number=''
            for i in range(L-len(str(id))):
                number+='0'
            number+=str(id)
            
            df.to_csv('./feature_tmp/0810_v2_g'+number+'/feature_0810_w1_g'+number+'_'+name+'.csv')
        