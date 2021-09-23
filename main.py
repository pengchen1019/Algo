import pandas as pd
import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

train_path = './data/train/'
valid_path = './data/validation/'

train_actions = pd.read_csv(train_path + 'train_actions_info.csv')
train_captcha = pd.read_csv(train_path + 'train_captcha_info.csv')
train_ground = pd.read_csv(train_path + 'train_ground_truth.csv')

valid_actions = pd.read_csv(valid_path + 'validation_actions_info.csv')
valid_captcha = pd.read_csv(valid_path + 'validation_captcha_info.csv')
valid_set = pd.read_csv(valid_path + 'validation_set.csv')

'''
Action_info(17565166 + 2478280):
    user_id:用户id
    action_id:用户行为id
    offset_hour:0表示1小时前的聚合数据，1表示1小时后的聚合数据
    pv:按前三项分组聚合后的请求次数

Captcha_info(304827 + 39836):
    user_id:用户id
    captcha_id:验证码id
    phase:验证阶段(new:验证生成   verify:用户验证)
    result:验证阶段结果(success   fail   error)
    verify_time:验证时间（毫秒）
    server_time:服务端回收验证码结果的时间戳(秒)

ground_truth(白:40048   黑:39980   共计80028条数据):
    user_id:用户id
    label:标签值(0:真人   1:机器人)

validation_set(10797):
    user_id:用户id
    type:A/B集标志，其中A:10%以下作弊，B:10%以下真人

验证方式：AB两样本集下召回率的乘积
'''

# 检测空值
# total = train_captcha.isnull().sum().sort_values(ascending=False)
# percent_1 = train_captcha.isnull().sum()/train_captcha.isnull().count()*100
# percent_2 = (round(percent_1,1)).sort_values(ascending=False)
# missing_data = pd.concat([total, percent_2],axis = 1, keys=['Total','%'])
# print(missing_data.head(5))

# 特征形式转换
'''
Captcha_info中，new：0 verify：1
                success：1 fail:0 error:-1
'''
phases = {'new': 0, 'verify': 1}
results = {'success': 1, 'fail': 0, 'error': -1}
dataset = [train_captcha, valid_captcha]
for data in dataset:
    data['phase'] = data['phase'].map(phases)
    data['result'] = data['result'].map(results)
