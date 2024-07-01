import configparser
import os
import re
import numpy as np
import pandas as pd

def get_config(key):
    config = configparser.ConfigParser()
    config.read(os.path.dirname(__file__) + '/config.ini')

    if 'config' not in config:
        raise Exception("Section 'config' not found in config.ini.")
        
    return config.get('config', key)

# Read data train
def read_data_train():
    df = pd.read_csv(get_config('file_train'), encoding='ISO-8859-1')
    return df

# Read data test
def read_data_test():
    df = pd.read_csv(get_config('file_test'), encoding='ISO-8859-1')
    return df

# Preprocessing Text
def clean(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Conver to lower
    text = text.lower()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__": 
    # trainning
    print(os.path.dirname(__file__))
