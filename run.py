import os
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
import matplotlib.pyplot as plt
import numpy as np
import smtplib, ssl
import os
from email.message import EmailMessage
##CUDA_VISIBLE_DEVICES=""

def analysis():
  
  cow = ""
  filepath = 'c:/Users/Sumeet/Desktop/CognitoMaster/saved_model'
  model = load_model(filepath, compile = True)
  model.save(filepath)

  one_file = pd.read_csv('./presses.csv')
  one_file.columns = ['key','press', 'duration', 'release']
  def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = [0]* dataframe.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
      ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds
  batch_size = 10
  run_ds = df_to_dataset(one_file, batch_size=batch_size)
  predictions = model.predict(run_ds)
  print(predictions)
  classes = np.argmax(predictions, axis = 1)
  print(classes)
  sum = 0
  for i in classes:
      sum = sum + i
  avg = sum/len(classes)
  if avg >= .5:
    cow = "We dected parkinsons from your typing. Please contact a doctor and seek help."
  else:
    cow = "We did not detect parkinsons. You are healthy!"
  ##load_dotenv()

  user_email = None
  with open('email.txt') as f:
      user_email = f.read()

  server = smtplib.SMTP('smtp.gmail.com', 587)
  #Next, log in to the server
  context = ssl.create_default_context()
  server.starttls(context=context)
  server.login("cognito.confirm@gmail.com", "Amazon!Mango!1234")

  #Send the mail
  msg = EmailMessage()
    
  msg.set_content("""
  Hey there! 
  {} 
  Thanks, 
  The Cognito Team""".format(cow))
  msg["Subject"] = "Cognito Email Confirm"
  msg["From"] = 'cognito.confirm@gmail.com'
  msg['To'] = user_email

    


  server.send_message(msg)
  server.quit()
analysis() 
