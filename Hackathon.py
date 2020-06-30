#Import statements
import numpy as np 
import pandas as pd 
import functools
import os, gc
import matplotlib.pyplot as plt
from itertools import chain
import sklearn.model_selection as skl
from random import sample 
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from keras.optimizers import Adam
import csv
from csv import writer
import glob
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras.models import Sequential, save_model, load_model
#Data preproccessing
user_file_list = os.listdir('../Hackathon4/Archived_Users/Archived users/')
user_set_v1 = set(map(lambda x: x[5: 15], user_file_list)) # [5: 15] to return just the user IDs


tappy_file_list = os.listdir('../Hackathon4/Archived_Data/Tappy Data/')
user_set_v2 = set(map(lambda x: x[: 10], tappy_file_list)) # [: 10] to return just the user IDs


user_set = user_set_v1.intersection(user_set_v2)

len(user_set)
def read_user_file(file_name):
    f = open('../Hackathon4/Archived_Users/Archived users/' + file_name)
    data = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

    return data
files = os.listdir('../Hackathon4/Archived_Users/Archived users/')

columns = [
    'BirthYear', 'Gender', 'Parkinsons', 'Tremors', 'DiagnosisYear',
    'Sided', 'UPDRS', 'Impact', 'Levadopa', 'DA', 'MAOB', 'Other'
]

user_df = pd.DataFrame(columns=columns) # empty Data Frame for now

for user_id in user_set:
    temp_file_name = 'User_' + user_id + '.txt' # tappy file names have the format of `User_[UserID].txt`
    if temp_file_name in files: # check to see if the user ID is in our valid user set
        temp_data = read_user_file(temp_file_name)
        user_df.loc[user_id] = temp_data # adding data to our DataFrame

user_df['BirthYear'] = pd.to_numeric(user_df['BirthYear'], errors='coerce')
user_df['DiagnosisYear'] = pd.to_numeric(user_df['DiagnosisYear'], errors='coerce')
user_df = user_df.rename(index=str, columns={'Gender': 'Female'}) # renaming `Gender` to `Female`
user_df['Female'] = user_df['Female'] == 'Female' # change string data to boolean data
user_df['Female'] = user_df['Female'].astype(int) # change boolean data to binary data
str_to_bin_columns = ['Parkinsons', 'Tremors', 'Levadopa', 'DA', 'MAOB', 'Other'] # columns to be converted to binary data

for column in str_to_bin_columns:
    user_df[column] = user_df[column] == 'True'
    user_df[column] = user_df[column].astype(int)
    # prior processing for `Impact` column
user_df.loc[
    (user_df['Impact'] != 'Medium') &
    (user_df['Impact'] != 'Mild') &
    (user_df['Impact'] != 'Severe'), 'Impact'] = 'None'

to_dummy_column_indices = ['Sided', 'UPDRS', 'Impact'] # columns to be one-hot encoded
for column in to_dummy_column_indices:
    user_df = pd.concat([
        user_df.iloc[:, : user_df.columns.get_loc(column)],
        pd.get_dummies(user_df[column], prefix=str(column)),
        user_df.iloc[:, user_df.columns.get_loc(column) + 1 :]
    ], axis=1)

user_df.head()
file_name = '0EA27ICBLF_1607.txt'
df = pd.read_csv(
    '../Hackathon4/Archived_Data/Tappy Data/' + file_name,
    delimiter = '\t',
    index_col = False,
    names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
)

df = df.drop('UserKey', axis=1)

print(df.head())
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date
# converting time data to numeric
for column in ['Hold time', 'Latency time', 'Flight time']:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df = df.dropna(axis=0)
# cleaning data in Hand
df = df[
    (df['Hand'] == 'L') |
    (df['Hand'] == 'R') |
    (df['Hand'] == 'S')
]

# cleaning data in Direction
df = df[
    (df['Direction'] == 'LL') |
    (df['Direction'] == 'LR') |
    (df['Direction'] == 'LS') |
    (df['Direction'] == 'RL') |
    (df['Direction'] == 'RR') |
    (df['Direction'] == 'RS') |
    (df['Direction'] == 'SL') |
    (df['Direction'] == 'SR') |
    (df['Direction'] == 'SS')
]
direction_group_df = df.groupby('Direction').mean()
direction_group_df
def read_tappy(file_name):
    df = pd.read_csv(
        '../Hackathon4/Archived-Data/Tappy Data/' + file_name,
        delimiter = '\t',
        index_col = False,
        names = ['UserKey', 'Date', 'Timestamp', 'Hand', 'Hold time', 'Direction', 'Latency time', 'Flight time']
    )

    df = df.drop('UserKey', axis=1)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%y%M%d').dt.date

    # converting time data to numeric
    #print(df[df['Hold time'] == '0105.0EA27ICBLF']) # for 0EA27ICBLF_1607.txt
    for column in ['Hold time', 'Latency time', 'Flight time']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(axis=0)

    # cleaning data in Hand
    df = df[
        (df['Hand'] == 'L') |
        (df['Hand'] == 'R') |
        (df['Hand'] == 'S')
    ]

    # cleaning data in Direction
    df = df[
        (df['Direction'] == 'LL') |
        (df['Direction'] == 'LR') |
        (df['Direction'] == 'LS') |
        (df['Direction'] == 'RL') |
        (df['Direction'] == 'RR') |
        (df['Direction'] == 'RS') |
        (df['Direction'] == 'SL') |
        (df['Direction'] == 'SR') |
        (df['Direction'] == 'SS')
    ]

    direction_group_df = df.groupby('Direction').mean()
    del df; gc.collect()
    direction_group_df = direction_group_df.reindex(['LL', 'LR', 'LS', 'RL', 'RR', 'RS', 'SL', 'SR', 'SS'])
    direction_group_df = direction_group_df.sort_index() # to ensure correct order of data
    
    return direction_group_df.values.flatten() # returning a numppy array

file_name = '0EA27ICBLF_1607.txt' # an arbitrary file to explore
tappy_data = read_tappy(file_name)

tappy_data

def process_user(user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.nanmean(running_user_data, axis=0) # ignoring NaNs while calculating the mean


filenames = os.listdir('../Hackathon4/Archived_Data/Tappy Data/')

user_id = '0EA27ICBLF'
process_user(user_id, filenames)

column_names = [first_hand + second_hand + '_' + time for first_hand in ['L', 'R', 'S'] for second_hand in ['L', 'R', 'S'] for time in ['Hold_time', 'Latency_time', 'Flight_time']]
print(column_names)
user_tappy_df = pd.DataFrame(columns=column_names)

for user_id in user_df.index:
    user_tappy_data = process_user(str(user_id), filenames)
    user_tappy_df.loc[user_id] = user_tappy_data

# some preliminary data cleaning
user_tappy_df = user_tappy_df.fillna(0)
user_tappy_df[user_tappy_df < 0] = 0    

user_tappy_df.head()

combined_user_df = pd.concat([user_df, user_tappy_df], axis=1)





combined_user_df['class'] = combined_user_df['Parkinsons']
del combined_user_df['Parkinsons']
del combined_user_df['DiagnosisYear']
del combined_user_df['UPDRS_1']
del combined_user_df['UPDRS_2']
del combined_user_df['UPDRS_3']
del combined_user_df['UPDRS_4']
del combined_user_df['Impact_Medium']
del combined_user_df['Impact_Mild']
del combined_user_df['Impact_None']
del combined_user_df['BirthYear']
del combined_user_df['Tremors']
del combined_user_df['Impact_Severe']
del combined_user_df['Levadopa']
del combined_user_df['DA']
del combined_user_df['MAOB']
del combined_user_df['Other']
del combined_user_df['UPDRS_Don\'t know']
combined_user_df['Female']=combined_user_df['Female'].astype(float)
combined_user_df['Sided_Left']=combined_user_df['Sided_Left'].astype(float)
combined_user_df['Sided_Right']=combined_user_df['Sided_Right'].astype(float)
combined_user_df['Sided_None']=combined_user_df['Sided_None'].astype(float)
combined_user_df['class']=combined_user_df['class'].astype(float)
print(combined_user_df.head(100))
train, test = train_test_split(combined_user_df, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
example_batch = next(iter(train_ds))[0]
def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

feature_columns = []

print(combined_user_df.columns)
print(combined_user_df.dtypes)
print(combined_user_df.isnull().any())
for header in ['Female', 'Sided_Left',
       'Sided_None', 'Sided_Right','LL_Hold_time', 
       'LL_Latency_time', 'LL_Flight_time', 'LR_Hold_time', 'LR_Latency_time',
       'LR_Flight_time', 'LS_Hold_time', 'LS_Latency_time', 'LS_Flight_time',
       'RL_Hold_time', 'RL_Latency_time', 'RL_Flight_time', 'RR_Hold_time',
       'RR_Latency_time', 'RR_Flight_time', 'RS_Hold_time', 'RS_Latency_time',
       'RS_Flight_time', 'SL_Hold_time', 'SL_Latency_time', 'SL_Flight_time',
       'SR_Hold_time', 'SR_Latency_time', 'SR_Flight_time', 'SS_Hold_time',
       'SS_Latency_time', 'SS_Flight_time']:
    print(feature_column.numeric_column(header))
    feature_columns.append(feature_column.numeric_column(header))
print(feature_columns)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential()
model.add(feature_layer)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))
model.add(Dense(1))


opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt,loss = "BinaryCrossentropy", metrics=['accuracy'])

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
model.fit(train_ds, validation_data = val_ds, epochs = 100, callbacks = [cp_callback] )
filepath = './saved_model2'
save_model(model, filepath, overwrite=True)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy:", accuracy)

