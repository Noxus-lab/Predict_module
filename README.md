# Stock Prediction Using LSTM

This repository contains code for predicting stock prices using LSTM (Long Short-Term Memory) neural networks.

## Overview

This project aims to predict stock prices using historical data and LSTM neural networks. It includes various data preprocessing steps, model training, and prediction writing to CSV files. The code supports batch processing for multiple sub-folders containing CSV files with historical stock data.

## Prerequisites

- Python 3.x
- Required libraries (install using `pip install -r requirements.txt`):
  - TensorFlow
  - numpy
  - yfinance
  - stockstats
  - scikit-learn
  - pandas
  - matplotlib

## Usage

1. Clone this repository:

2. Install the required libraries:


3. Adjust the command-line arguments in the `main.py` script to match your needs.

4. Run the script:


## Directory Structure

- `main.py`: The main script to run the stock prediction process.
- `Feature-Pool/`: Sample historical stock data in CSV format.
- `output/`: Output directory for individual prediction CSV files.
- `merge/`: Output directory for merged prediction CSV files.


## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [stockstats](https://pypi.org/project/stockstats/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)


