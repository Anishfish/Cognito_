import os, gc
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
import matplotlib.pyplot as plt
import numpy as np

user_file_list = os.listdir('../Hackathon4/Archived-Users/Archived users/')
user_set_v1 = set(map(lambda x: x[5: 15], user_file_list))
tappy_file_list = os.listdir('../Hackathon4/Archived_Data/Tappy Data/')
user_set_v2 = set(map(lambda x: x[: 10], tappy_file_list))
def read_user_file(file_name):
    f = open('../Hackathon4/Archived_Users/Archived users/' + file_name)
    data = [line.split(': ')[1][: -1] for line in f.readlines()]
    f.close()

    return data
files = os.listdir('../Hackathon4/Archived-users/Archived users/')
user_set = user_set_v1.intersection(user_set_v2)
columns =['Gender','Sided']
user_df = pd.DataFrame(columns=columns)
user_df = user_df.rename(index=str, columns={'Gender': 'Female'}) # renaming `Gender` to `Female`
user_df['Female'] = user_df['Female'] == 'Female' # change string data to boolean data
user_df['Female'] = user_df['Female'].astype(int) 

user_df.head()
file_name = 'presses.txt'
df = pd.read_csv(
    '../Hackathon4/Archived-Data/Tappy Data/' + file_name,
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

file_name = 'presses.txt' # an arbitrary file to explore
tappy_data = read_tappy(file_name)

tappy_data

def process_user(user_id, filenames):
    running_user_data = np.array([])

    for filename in filenames:
        if user_id in filename:
            running_user_data = np.append(running_user_data, read_tappy(filename))
    
    running_user_data = np.reshape(running_user_data, (-1, 27))
    return np.nanmean(running_user_data, axis=0)

filenames = os.listdir('../Hackathon4/Archived-Data/Tappy Data/')
user_id = '. Patient1234'
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


combined_user_df['Female']=combined_user_df['Female'].astype(float)
combined_user_df['Sided_Left']=combined_user_df['Sided_Left'].astype(float)
combined_user_df['Sided_Right']=combined_user_df['Sided_Right'].astype(float)
combined_user_df['Sided_None']=combined_user_df['Sided_None'].astype(float)
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 10
run_ds = df_to_dataset(combined_user_df, batch_size=batch_size)






predictions = model.predict(run_ds)
print(predictions)
classes = np.argmax(predictions, axis = 1)
print(classes)
