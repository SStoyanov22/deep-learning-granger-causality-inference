# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:03:45 2022

@author: stoyan.stoyanov
"""
#%%
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math # Mathematical functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import  MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns

class lstm_model:

    def __init__(self,df,target):
        self.df = df
        self.target = target
        self.unscaled_data = self.data_prepare(df)
        self.scaler = MinMaxScaler()
        self.scaled_data =  self.scaler.fit_transform(self.unscaled_data)
        self.scaler_pred = MinMaxScaler()
        self.scaled_pred_data =  self.scaler_pred.fit_transform(pd.DataFrame(self.unscaled_data[0]))
        self.x_train, self.y_train, self.x_test, self.y_test = self.split()
        self.model = self.create()

    def data_prepare(self,df):
        # Convert the data to numpy values
        np_data_unscaled = np.array(df)
        
        return np_data_unscaled

    def partition_dataset(self,sequence_length, data):
        # Prediction Index
        index_target = self.df.columns.get_loc(self.target)
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columns
            y.append(data[i, index_target]) #contains the prediction values for validation,  for single-step prediction
        
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y
    
    def split(self):
        # Set the sequence length - this is the timeframe used to make a single prediction
        sequence_length = 50
        
        # Split the training data into train and train data sets
        # As a first step, we get the number of rows to train the model on 80% of the data 
        train_data_len = math.ceil(self.scaled_data.shape[0] * 0.8)
    
        # Create the training and test data
        train_data = self.scaled_data[0:train_data_len, :]
        test_data = self.scaled_data[train_data_len - sequence_length:, :]
        
        # Generate training data and test data
        x_train, y_train = self.partition_dataset(sequence_length, train_data)
        x_test, y_test = self.partition_dataset(sequence_length, test_data)
        
        return x_train, y_train, x_test, y_test
    
    
    def create(self):
        # Configure the neural network model
        model = Sequential()
        # Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
        n_neurons = self.x_train.shape[1] * self.x_train.shape[2]
        
        model.add(LSTM(n_neurons, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2]))) 
        model.add(LSTM(n_neurons, return_sequences=False))
        model.add(Dense(5))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def train(self):
        # Training the model
        epochs = 50
        batch_size = 16
        #early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        history = self.model.fit(self.x_train, self.y_train, 
                            batch_size=batch_size, 
                            epochs=epochs,
                            validation_data=(self.x_test,self.y_test)
                           )
                        
                        #callbacks=[early_stop])
        # Plot training & validation loss values
        fig, ax = plt.subplots(figsize=(20, 10), sharex=True)
        plt.plot(history.history["loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
        plt.legend(["Train", "Test"], loc="upper left")
        plt.grid()
        plt.show()
