import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot  as plt
import seaborn as sns


import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

dataset = pd.read_csv("/content/drive/MyDrive/BigData/04_hashed.csv")

dataset.columns
dataset.head()

# def select_unique_Rdate(dataset):
#   time_list=[]
#   for i in range(len(dataset['Rdate'])):
#     time_list.append(dataset.iloc[i]['Rdate'])
#   unique_time = set(time_list)
#   unique_time = sorted(unique_time)
#   print(len(unique_time)) # 24218
#   return unique_time 
start_time = 20210411000000
get_days = 1000000

dataset_origin = dataset

dataset = dataset[dataset['Rdate'] < 20210411000000 ]

dataset.tail()

dataset.loc[2861154]['Rdate']


# feature = dataset[ ['src_ip', 'dst_ip', 'Proto', 'src_port', 'dst_port', 'Action',
#        'src_country', 'dst_country'] ]

feature = dataset[['Rdate','src_ip', 'dst_ip', 'src_port', 'dst_port', 'Action']]

# IPv4 전처리
def transform_ip(ip): 
  groups = ip.split(".") 
  equalize_group_length = "".join( map( lambda group: group.zfill(3), groups )) 
  return equalize_group_length 

from sklearn.preprocessing import LabelEncoder

# Feature 전처리
def preprocess_df(df):
  
  # IPv4 전처리
  df['src_ip'] = df.src_ip.apply(lambda ip : transform_ip(ip))
  df['dst_ip'] = df.dst_ip.apply(lambda ip : transform_ip(ip))

  # country 전처리
  label_encoder = LabelEncoder()
  df['src_country'] = label_encoder.fit_transform(df['src_country'])
  df['dst_country'] = label_encoder.fit_transform(df['dst_country'])

  return df

dataset = preprocess_df(dataset)

# #############################################################################
# Compute DBSCAN
def dbscan(feature_):
  from sklearn.preprocessing import StandardScaler
  feature = feature_[ ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'Action']]

  scaler = StandardScaler()

  scaler.fit(feature)
  feature_trans = scaler.transform(feature)
  model = DBSCAN(eps=0.5, min_samples=5)
  predict = pd.DataFrame(model.fit_predict(feature_trans))
  predict.columns = ['predict']
  r = pd.concat([feature,predict],axis=1)
  
  # core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
  # core_samples_mask[model.core_sample_indices_] = True
  labels = model.labels_

  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise_ = list(labels).count(-1)

  print("Dataset Size : ",len(feature))
  print("Estimated number of clusters: %d" % n_clusters_)
  print("Estimated number of noise points: %d" % n_noise_)
  return r, labels

def pair_plot(r):
  sns.pairplot(r,hue='predict')
  plt.show()


# window size 만큼의의 데이터 추출
dbscan_result = dbscan(dataset)
pair_plot(dbscan_result)


anomaly_data = dbscan_result[dbscan_result['predict']==-1]

anomaly_data.to_csv("/content/drive/MyDrive/dbscan_anomaly_data.csv")


