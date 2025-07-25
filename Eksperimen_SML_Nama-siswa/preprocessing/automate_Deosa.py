# -*- coding: utf-8 -*-
"""automate_hafis_afrizal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SJnGtQEDwvGK83j7-TDdNM2wQlkvCrX-
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def preprocess_data(input_path, output_path):
    # Pastikan direktori output ada
    os.makedirs(output_path, exist_ok=True)

    # Muat dataset
    df = pd.read_csv(input_path)

    # 1. Hapus kolom tidak perlu
    df = df.drop(columns=['Id'], errors='ignore')

    # 2. Encoding target
    le = LabelEncoder()
    df['Species'] = le.fit_transform(df['Species'])

    # 3. Pisahkan fitur dan target
    X = df.drop(columns=['Species'])
    y = df['Species']

    # 4. Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Bagi data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 6. Simpan hasil
    pd.DataFrame(X_train).to_csv(f'{output_path}/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv(f'{output_path}/X_test.csv', index=False)
    pd.DataFrame(y_train).to_csv(f'{output_path}/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{output_path}/y_test.csv', index=False)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    input_path = "Iris.csv"
    output_path = "preprocessing/namadataset_preprocessing"
    preprocess_data(input_path, output_path)