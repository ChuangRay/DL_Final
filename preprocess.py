import pandas as pd
import numpy as np
from tqdm import tqdm

# 顯示所有行列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

train = pd.read_csv('/Users/chuangray/Desktop/DL_Final/train.csv', header=0, low_memory=False)
test = pd.read_csv('/Users/chuangray/Desktop/DL_Final/test.csv', header=0, low_memory=False)

# -------------------------------------------------------前處理-------------------------------------------------------
def checkNaN(data, fill):
    print(data.name + ' nan = ', data.isna().sum())
    if data.isna().sum() != 0:
        if fill is not None:
            data = data.fillna(fill)
        else:
            data = data.fillna(data.mean())
        print(data.name + ' 補值後nan = ', data.isna().sum())
    return data


def standardization(data):
    data = (data - data.mean()) / data.std()
    print(data.describe())
    print()
    return data


df_feature = train
# 酪農場代號 one hot
farm_onehot = pd.get_dummies(df_feature['酪農場代號'], prefix='酪農場代號')
df_feature = df_feature.drop('酪農場代號', axis=1)
df_feature = pd.concat([df_feature, farm_onehot], axis=1)

# 乳牛編號
data = np.unique(df_feature['乳牛編號'])
index = np.arange(len(data))
cow_id = zip(data, index+1)
cowidDict = dict(cow_id)
df_feature['乳牛編號'] = standardization(df_feature['乳牛編號'].map(cowidDict))

# 出生日期 -> 季節 -> one hot
print('出生日期.nan = ', df_feature['出生日期'].isna().sum())
seasonDict = {3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter', 1: 'Winter', 2: 'Winter'}
df_feature['出生日期'] = pd.to_datetime(df_feature['出生日期']).dt.month
df_feature['出生日期'] = df_feature['出生日期'].map(seasonDict)
print(df_feature['出生日期'].head(n=5))
season_onehot = pd.get_dummies(df_feature['出生日期'], prefix='出生日期')
df_feature = df_feature.drop('出生日期', axis=1)
df_feature = pd.concat([df_feature, season_onehot], axis=1)

# 胎次 Standardization
df_feature['胎次'] = checkNaN(df_feature['胎次'], fill=None)
df_feature['胎次'] = standardization(df_feature['胎次'])

# 泌乳天數 Standardization
df_feature['泌乳天數'] = checkNaN(df_feature['泌乳天數'], fill=None)
df_feature['泌乳天數'] = standardization(df_feature['泌乳天數'])

# 距離最近分娩天數 (採樣日期 - 最近分娩日期) Standardization
df_feature['距離最近分娩天數'] = pd.to_datetime(df_feature['採樣日期']) - pd.to_datetime(df_feature['最近分娩日期'])
df_feature['距離最近分娩天數'] = checkNaN(df_feature['距離最近分娩天數'], fill=None)
df_feature['距離最近分娩天數'] = standardization(df_feature['距離最近分娩天數'])
print(df_feature['距離最近分娩天數'].head(n=5))

# 月齡 Standardization
df_feature['月齡'] = checkNaN(df_feature['月齡'], fill=None)
df_feature['月齡'] = standardization(df_feature['月齡'])

# 配種次數 Standardization
df_feature['配種次數'] = checkNaN(df_feature['配種次數'], fill=None)
df_feature['配種次數'] = standardization(df_feature['配種次數'])

# 兩次分娩間隔 (最近分娩日期 - 前次分娩日期) Standardization
df_feature['兩次分娩間隔'] = pd.to_datetime(df_feature['最近分娩日期']) - pd.to_datetime(df_feature['前次分娩日期'])
df_feature['兩次分娩間隔'] = checkNaN(df_feature['兩次分娩間隔'], fill='0')
df_feature['兩次分娩間隔'] = standardization(df_feature['兩次分娩間隔'])

# 母牛體重 Standardization
df_feature['母牛體重'] = checkNaN(df_feature['母牛體重'], fill=None)
df_feature['母牛體重'] = standardization(df_feature['母牛體重'])

# 分娩難易度 Standardization
df_feature['分娩難易度'] = checkNaN(df_feature['分娩難易度'], fill=1)
df_feature['分娩難易度'] = standardization(df_feature['分娩難易度'])

# 犢牛性別  one hot(可刪)
df_feature['犢牛性別'] = checkNaN(df_feature['犢牛性別'], fill='unknow')
gender_onehot = pd.get_dummies(df_feature['犢牛性別'], prefix='犢牛性別')
df_feature = df_feature.drop('犢牛性別', axis=1)
df_feature = pd.concat([df_feature, gender_onehot], axis=1)

# drop 乾乳日期(和泌乳天數重疊)
# drop 計算胎次(全部都為True)
# drop 胎次(birth)(和birth相同)





