'''
# colab
from google.colab import drive
drive.mount('/content/drive')
!ls /content/drive/
%cd "/content/drive/My Drive"
%cd "Colab Notebooks/nthudl/final_project"
!ls
'''

import pandas as pd
import numpy as np
import re

# read csv files
df_report = pd.read_csv('data/report.csv',header=0,names=[
            "ID","資料年度","資料月份","酪農場代號","乳牛編號",
            "父親牛精液編號","母親乳牛編號","出生日期","胎次","泌乳天數",
            "乳量","最近分娩日期","採樣日期","月齡","檢測日期",
            "最後配種日期","最後配種精液","配種次數","前次分娩日期","第一次配種日期","第一次配種精液"])
df_birth = pd.read_csv('data/birth.csv',header=0,names=[
            "乳牛編號","分娩日期","乾乳日期","犢牛編號1","讀牛編號2",
            "母牛體重","登錄日期","計算胎次","胎次","分娩難易度",
            "犢牛體型","犢牛性別","酪農場代號"])
df_breed = pd.read_csv('data/breed.csv',header=0,names=[
            "乳牛編號","配種日期","配種精液","登錄日期","孕檢",
            "配種方式","精液種類","酪農場代號"])
df_spec = pd.read_csv('data/spec.csv',header=0,names=[
            "乳牛編號","狀況類別","裝況代號","狀況日期","備註",
            "登錄日期","酪農場代號"])

'''
# show report
df_report
'''

# 量化
def csv_preprocess(df):
  # nan -> 0
  df = df.fillna(0)

  # 酪農場代號量化
  placeDict={'A':0,'B':1,'C':2}
  df["酪農場代號"] = df["酪農場代號"].map(placeDict)

  # 乳牛編號
  data = np.unique(df['乳牛編號'])
  index = np.arange(len(data))
  cow_id = zip(data,index)
  cowidDict = dict(cow_id) 
  # print(cowidDict)
  df['乳牛編號'] = df['乳牛編號'].map(cowidDict)

  # 父親牛精液編號
  df['父親牛精液編號'] = df['父親牛精液編號'].str.replace('[A-Z]','')
  df['父親牛精液編號'] = df['父親牛精液編號'].str.replace('進口','1')
  df['父親牛精液編號'] = df['父親牛精液編號'].str.replace('外購','2')
  df['父親牛精液編號'] = df['父親牛精液編號'].fillna(0).astype(int)
  daddata = np.unique(df['父親牛精液編號'])
  dadindex = np.arange(len(daddata))
  dadcow_id = zip(daddata,dadindex)
  dadcowidDict = dict(dadcow_id) 
  print(dadcowidDict)
  df['父親牛精液編號'] = df['父親牛精液編號'].map(dadcowidDict)

  # 母親乳牛編號
  df['母親乳牛編號'] = df['母親乳牛編號'].str.replace('[A-Z]','')
  df['母親乳牛編號'] = df['母親乳牛編號'].str.replace('進口','1')
  df['母親乳牛編號'] = df['母親乳牛編號'].str.replace('外購','2')
  df['母親乳牛編號'] = df['母親乳牛編號'].fillna(0).astype(int)
  momdata = np.unique(df['母親乳牛編號'])
  momindex = np.arange(len(momdata))
  momcow_id = zip(momdata,momindex)
  momcowidDict = dict(momcow_id) 
  print(momcowidDict)
  df['母親乳牛編號'] = df['母親乳牛編號'].map(momcowidDict)

  # 合併配種精液的Dictionary
  
  # 最後配種精液
  df['最後配種精液'] = df['最後配種精液'].str.replace('[A-Z]','0')
  df['最後配種精液'] = df['最後配種精液'].fillna(0).astype(int)
  # lastdata = np.unique(df['最後配種精液'])
  # lastindex = np.arange(len(lastdata))
  # lastcow_id = zip(lastdata,lastindex)
  # lastcowidDict = dict(lastcow_id) 
  # print(lastcowidDict)
  # df['最後配種精液'] = df['最後配種精液'].map(lastcowidDict)

  # 第一次配種精液
  df['第一次配種精液'] = df['第一次配種精液'].str.replace('[A-Z]','0')
  df['第一次配種精液'] = df['第一次配種精液'].fillna(0).astype(int)
  # firstdata = np.unique(df['第一次配種精液'])
  # firstindex = np.arange(len(firstdata))
  # firstcow_id = zip(firstdata,firstindex)
  # firstcowidDict = dict(firstcow_id) 
  # print(firstcowidDict)
  # df['第一次配種精液'] = df['第一次配種精液'].map(firstcowidDict)
  temp = np.unique([df['最後配種精液'],df['第一次配種精液']])
  tempindex = np.arange(len(temp))
  tempcow_id = zip(temp,tempindex)
  tempcowidDict = dict(tempcow_id) 
  print(tempcowidDict)
  df['最後配種精液'] = df['最後配種精液'].map(tempcowidDict)
  df['第一次配種精液'] = df['第一次配種精液'].map(tempcowidDict)
  

  return df

df_re = csv_preprocess(df_report)

# 捨去0很多的column
def drop(df,col_names):

  if np.sum((df[col_names]==0)) >= 0.5*len(df[col_names]):
    print('[{}]: {}/{}'.format(col_names,np.sum(df[col_names]==0),0.5*len(df[col_names])))
    df.drop(col_names)
  print('[{}]: {}/{}'.format(col_names,np.sum(df[col_names]==0),0.5*len(df[col_names])))
  return df

for col_names in df_re.columns:
  df_re = drop(df_re,col_names)