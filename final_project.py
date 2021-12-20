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
# 顯示所有行列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth',100)

# read csv files
df_report = pd.read_csv('/Users/chuangray/Desktop/DL_Final/data/report.csv',header=0, low_memory=False,
                        names=["ID","資料年度","資料月份","酪農場代號","乳牛編號",
                                "父親牛精液編號","母親乳牛編號","出生日期","胎次","泌乳天數",
                                "乳量","最近分娩日期","採樣日期","月齡","檢測日期",
                                "最後配種日期","最後配種精液","配種次數","前次分娩日期","第一次配種日期","第一次配種精液"])
df_birth = pd.read_csv('/Users/chuangray/Desktop/DL_Final/data/birth.csv',header=0, low_memory=False,
                        names=["乳牛編號","分娩日期","乾乳日期","犢牛編號1","讀牛編號2",
                               "母牛體重","登錄日期","計算胎次","胎次","分娩難易度",
                               "犢牛體型","犢牛性別","酪農場代號"])
df_breed = pd.read_csv('/Users/chuangray/Desktop/DL_Final/data/breed.csv',header=0, low_memory=False,
                        names=["乳牛編號","配種日期","配種精液","登錄日期","孕檢",
                               "配種方式","精液種類","酪農場代號"])
df_spec = pd.read_csv('/Users/chuangray/Desktop/DL_Final/data/spec.csv',header=0, low_memory=False,
                        names=["乳牛編號","狀況類別","狀況代號","狀況日期","備註",
                               "登錄日期","酪農場代號"])
df_submission = pd.read_csv('/Users/chuangray/Desktop/DL_Final/data/submission.csv',header=0, low_memory=False)

print("df_report shape = ", df_report.shape)
print("df_birth shape = ", df_birth.shape)
print("df_breed shape = ", df_breed.shape)
print("df_spec shape = ", df_spec.shape)
print("df_submission shape = ", df_submission.shape)

# -----------------------------------------------------合併資料-----------------------------------------------------
# 約執行7分鐘
'''
# create feature table
birth_feature = pd.DataFrame(columns=["乾乳日期", "母牛體重", "計算胎次", "胎次", "分娩難易度", "犢牛體型","犢牛性別"],
                             index=df_report.index)
breed_feature = pd.DataFrame(columns=["配種日期","配種精液","孕檢","配種方式","精液種類"],
                             index=df_report.index)
spec_feature = pd.DataFrame(columns=["狀況類別","狀況代號","狀況日期"],
                             index=df_report.index)

for idx in tqdm(range(0, len(df_report))):
    # ---------------------------------birth---------------------------------
    mask1 = df_birth['乳牛編號'] == df_report.at[idx, '乳牛編號']
    mask2 = df_birth['分娩日期'] == df_report.at[idx, '最近分娩日期']
    a = df_birth[mask1 & mask2]
    if df_birth[mask1 & mask2].empty == False:
        birth_feature.loc[df_report.index[idx], ["乾乳日期", "母牛體重", "計算胎次", "胎次", "分娩難易度", "犢牛體型","犢牛性別"]] = \
            a.iloc[0,2], a.iloc[0,5], a.iloc[0,7], a.iloc[0,8], a.iloc[0,9], a.iloc[0,10], a.iloc[0,11]
    else:
        birth_feature.loc[df_report.index[idx], ["乾乳日期", "母牛體重", "計算胎次", "胎次", "分娩難易度", "犢牛體型","犢牛性別"]] = \
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # ---------------------------------breed---------------------------------
    mask3 = df_breed['乳牛編號'] == df_report.at[idx, '乳牛編號']
    breed = df_breed[mask3]['配種日期']

    for i in range(0, len(breed)):
        if (pd.to_datetime(breed.iloc[i]) > pd.to_datetime(df_report.at[idx, '前次分娩日期'])) & \
                (pd.to_datetime(breed.iloc[i]) <= pd.to_datetime(df_report.at[idx, '採樣日期'])):
            breed_feature.loc[df_report.index[idx], ["配種日期","配種精液","孕檢","配種方式","精液種類"]] = \
                df_breed[mask3].iloc[i,1], df_breed[mask3].iloc[i,2], df_breed[mask3].iloc[i,4], \
                df_breed[mask3].iloc[i,5], df_breed[mask3].iloc[i,6]
        else:
            breed_feature.loc[df_report.index[idx], ["配種日期","配種精液","孕檢","配種方式","精液種類"]] = \
                np.nan, np.nan, np.nan, np.nan, np.nan

    # ---------------------------------spec---------------------------------
    mask4 = df_spec['乳牛編號'] == df_report.at[idx, '乳牛編號']
    spec = df_spec[mask4]['狀況日期']
    for j in range(0, len(spec)):
        if (pd.to_datetime(spec.iloc[j]) > pd.to_datetime(df_report.at[idx, '前次分娩日期'])) & \
                (pd.to_datetime(spec.iloc[j]) <= pd.to_datetime(df_report.at[idx, '最近分娩日期'])):
            spec_feature.loc[df_report.index[idx], ["狀況類別","狀況代號","狀況日期"]] = \
                df_spec[mask4].iloc[j,1], df_spec[mask4].iloc[j,2], df_spec[mask4].iloc[j,3]
        else:
            spec_feature.loc[df_report.index[idx], ["狀況類別","狀況代號","狀況日期"]] = np.nan, np.nan, np.nan


print(birth_feature.head(n=10))
print(breed_feature.head(n=10))
print(spec_feature.head(n=10))
print(birth_feature.shape)
print(breed_feature.shape)
print(spec_feature.shape)

FeatureTable = pd.concat([df_report.iloc[:,0:14], df_report.iloc[:,17:19]], axis=1, join='inner')
FeatureTable = pd.concat([FeatureTable, birth_feature], axis=1, join='inner')
FeatureTable = pd.concat([FeatureTable, breed_feature], axis=1, join='inner')
FeatureTable = pd.concat([FeatureTable, spec_feature], axis=1, join='inner')
print(FeatureTable.shape)

# 匯出成csv
def output():
    filepath = '/Users/chuangray/Desktop/DL_Final/Feature.csv'
    df_SAMPLE = pd.DataFrame.from_dict(FeatureTable)
    df_SAMPLE.to_csv(filepath, index=False)
    print('Success output to ' + filepath)

output()
'''

# ----------------------------------------------將submission的資料拆分出來----------------------------------------------
# 約執行1分鐘
df_feature = pd.read_csv('/Users/chuangray/Desktop/DL_Final/Feature.csv',header=0, low_memory=False)

dataset = pd.DataFrame()
submission_dataset = pd.DataFrame()
for idx in tqdm(range(0, len(df_submission))):
    mask = df_feature['ID'] == df_submission.at[idx, 'ID']
    submission_dataset = pd.concat([submission_dataset, df_feature[mask]], ignore_index=True)
    df_feature.drop(df_feature[mask].index, inplace=True)

print(df_feature.shape)
print(submission_dataset.shape)
=======
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
>>>>>>> master
>>>>>>> Stashed changes
