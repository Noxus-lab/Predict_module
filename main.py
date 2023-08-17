from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import yfinance as yf
import stockstats
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define function to parse command-line arguments
def get_args():
    """
    Parse command-line arguments using ArgumentParser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='0810_v1', help='Input_Folder')
    parser.add_argument('--data_path', type=str, default='./Feature-Pool/', help='Input Main Folder.')
    parser.add_argument('--batch_num', type=int, default=32, help='number of batchs')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='number of epochs')
    parser.add_argument('--time_window', type=int, default=10, help='time_window')
    parser.add_argument('--output_folder', type=str, default='0817_v1', help='time_window')
    return parser.parse_args()

# Define function to get all sub-folders that match a given prefix
def get_all_group(folder_path,target_folder):
    """
    Get a list of all sub-folders within the specified folder path that match a given prefix.

    Args:
        folder_path (str): Path to the main folder.
        target_folder (str): Prefix of sub-folders to match.

    Returns:
        list: List of sub-folder names matching the prefix, sorted.
    """

    sub_folders_g = [sub_folder_name for sub_folder_name in os.listdir(folder_path)
               if os.path.isdir(os.path.join(folder_path, sub_folder_name))
               and sub_folder_name.startswith(target_folder)]
    sub_folders_g = sorted(sub_folders_g)
    return sub_folders_g



# Define a function to prepare data for training
def prepare_data(n, csv_file, sub_folder_path):
    """
    Prepare data for training.

    Args:
        n (int): Time window size.
        csv_file (str): CSV file name.
        sub_folder_path (str): Path to the sub-folder containing the CSV file.

    Returns:
        tuple: Tuple containing training and testing data arrays.
    """

    #load data from csv files
    file_path = os.path.join(sub_folder_path, csv_file)
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    columns_to_drop = ["SmaDiffWeekMonth","IsBiggestVolume","SeasonVolatility"]
    df = df.drop(columns_to_drop, axis=1)

    #Transform data frame in to the defined form 
    frameNum = 6
    array_3d = np.empty((0,df.shape[0], frameNum), dtype=np.int64)
    array_3d =  np.append(array_3d,[df.values],axis=0)
    transposed_array = np.swapaxes(array_3d, 0,1)

    original_array = transposed_array

    new_shape = (transposed_array.shape[0] - n, n, transposed_array.shape[1], frameNum)

    all_data = np.empty(new_shape)

    true_labels = original_array[(n ):, :,0]
    for i in range(new_shape[0]):
        all_data[i] = original_array[i:i+n, :, :]


    #produce labels and split origin data into train and test data set
    reshape_all_data = all_data.reshape(all_data.shape[0], n, -1)
    test_num = 100
    decay_num = 0
    train_data_num = 240-n-1
    diff_true_label = np.diff(np.squeeze(true_labels)).reshape(-1,1)
    diff_all_data = reshape_all_data[1:]
    train_data = diff_all_data[decay_num:train_data_num,:]
    train_label = diff_true_label[decay_num:train_data_num,:]
    test_data = diff_all_data[train_data_num:train_data_num+test_num,:]
    test_label = diff_true_label[train_data_num:train_data_num+test_num,:]

    #shuffle train data
    X_train = train_data
    y_train = train_label
    X_test = test_data
    y_test = test_label
    shuffle_indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    return X_train,y_train,X_test,y_test



# Define a function to build and train an LSTM model
def train_module(X_train, y_train, X_test, y_test, epoch, learning_rate, batch, output_file_name):
    """
    Build and train an LSTM model.

    Args:
        X_train (numpy.ndarray): Training input data.
        y_train (numpy.ndarray): Training target data.
        X_test (numpy.ndarray): Testing input data.
        y_test (numpy.ndarray): Testing target data.
        epoch (int): Number of training epochs.
        learning_rate (float): Learning rate for optimization.
        batch (int): Batch size.
        output_file_name (str): Name of the output file.

    Returns:
        numpy.ndarray: Model predictions on the testing data.
    """


    print(output_file_name)
    split_result = output_file_name.split("_")
    output_file_name = split_result[-1]

    #define model and train
    def build_lstm_model(timesteps, num_features):
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(128, input_shape=(timesteps, num_features)),
                    tf.keras.layers.Dense(1)
                ])
                model.build(input_shape=(None, timesteps, num_features))
                return model
            
    model = build_lstm_model(timesteps=X_train.shape[1], num_features=X_train.shape[2])

    model.compile(optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=epoch, batch_size=batch, validation_data=(X_test, y_test), verbose=0)


    #caculate correlation between train predictions and train labels
    y_pred = model.predict(X_train, verbose=0)
    true_temp = y_train
    pred_temp = y_pred
    correlation_matrix = np.corrcoef(true_temp.reshape(-1), pred_temp.reshape(-1))
    correlation_coefficient = correlation_matrix[0, 1]
    print("train:"+output_file_name+" :", correlation_coefficient)


    #caculate correlation between test predictions and tets labels
    y_pred = model.predict(X_test, verbose=0)
    true_temp = y_test
    pred_temp = y_pred
    correlation_matrix = np.corrcoef(true_temp.reshape(-1), pred_temp.reshape(-1))
    correlation_coefficient = correlation_matrix[0, 1]
    print("test:"+output_file_name+" :", correlation_coefficient)
    print("-----------------------------------")
    return y_pred




# Define a function to write predictions to a CSV file
def write_to_csv(pred_temp, output_file_name, file_path):
    """
    Write predictions to a CSV file.

    Args:
        pred_temp (numpy.ndarray): Predictions to write.
        output_file_name (str): Name of the output file.
        file_path (str): Path to the original CSV file.
    """

    split_result = output_file_name.split("_")
    output_file_name = split_result[-1]
    df_new = pd.DataFrame(pred_temp, columns=[output_file_name])
    df = pd.read_csv(file_path)
    all_col = ["SmaDiffWeekMonth","IsBiggestVolume","SeasonVolatility","WeekMomentum","MonthMomentum","ReturnVelocity","ReturnAcceleration","VolumeVelocity"]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.drop(all_col, axis=1)
    df = df.tail(pred_temp.shape[0])
    df_new['Date']=df.index
    df_new.set_index('Date', inplace=True)
    df_new.index = pd.to_datetime(df.index, utc=True)
    df_new.index = df_new.index.strftime('%Y-%m-%d')
    if not os.path.exists("./output/"):
            os.makedirs("./output/")
    df_new.to_csv("./output/"+output_file_name+".csv", index=True)


# Define function to merge CSV files into an output file
def merge_to_output(out_put_name, sub_folder_name):
    """
    Merge CSV files into an output file.

    Args:
        out_put_name (str): Name of the output file.
        sub_folder_name (str): Name of the sub-folder.
    """

    folder_path_m = './output/'
    csv_files = sorted([f for f in os.listdir(folder_path_m) if f.endswith('.csv')])
    dataframes = []

    #merge all the files in "./output" folder into one data frame
    for csv_file in csv_files:
        file_path = os.path.join(folder_path_m, csv_file)
        df_new = pd.read_csv(file_path)
        df_new = df_new.drop('Date', axis=1)
        dataframes.append(df_new)
        merged_df = pd.concat(dataframes, axis=1)

    #sotre the data into csv
    df = pd.read_csv("./output/"+"AAPL"+".csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    if not os.path.exists("./merge/"+out_put_name+"/"):
        os.makedirs("./merge/"+out_put_name+"/")
    merged_df.index = pd.to_datetime(df.index, utc=True)
    merged_df.index = merged_df.index.strftime('%Y-%m-%d')
    merged_df.to_csv("./merge/"+out_put_name+"/"+out_put_name+sub_folder_name[-4:]+".csv", index=True)


if __name__ == '__main__':
    args = get_args()

    #mute tf warring
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'  
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '5'  
    tf.get_logger().setLevel('ERROR')

    #get the folder for training and testing
    all_folder = get_all_group(args.data_path,args.input_file) #

    for sub_folder_name in all_folder:
        # concate args.data_path and sub_folder_name to create sub_folder_path
        sub_folder_path = os.path.join(args.data_path, sub_folder_name)
        if os.path.isdir(sub_folder_path) and sub_folder_name.startswith(args.input_file):

            # get and rearrange the order of incoming csv files
            csv_files = sorted([f for f in os.listdir(sub_folder_path) if f.endswith('.csv')])
            for csv_file in csv_files:
                file_path = os.path.join(sub_folder_path, csv_file)
                #get train and test data
                X_train,y_train,X_test,y_test = prepare_data(args.time_window,csv_file,sub_folder_path)
                #get prediction output
                y_pred = train_module(X_train,y_train,X_test,y_test,args.epoch,args.learning_rate,args.batch_num,csv_file[:-4])
                #write back output
                write_to_csv(y_pred,csv_file[:-4],file_path)
            #merge all the csv files in same group
            merge_to_output(args.output_folder ,sub_folder_name)