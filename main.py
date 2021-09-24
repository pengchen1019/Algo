import pandas as pd
import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
    phase:验证阶段(new:验证生成   verify:用户验证 filter：被前端豁免过滤，比如该用户刚刚通过了其他验证码的测试 view:验证码弹出展示)
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
Captcha_info中，new：0 verify：1   filter:2   view:3
                success：1 fail:2 error:0
'''
phases = {'new': 0, 'verify': 1, 'filter': 2, 'view': 3}
results = {'success': 1, 'fail': 2, 'error': 0}
dataset = [train_captcha, valid_captcha]
for data in dataset:
    data['phase'] = data['phase'].map(phases)
    data['result'] = data['result'].map(results)

#相关性热力图绘制
# data_corr = train_captcha.corr()
# plt.subplots(figsize=(9, 9), dpi=100, facecolor='w')
# fig = sns.heatmap(data_corr, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
# fig.set_title('Train_captcha相关性热力图')
# plt.show()

#总平均请求次数
# merge = (valid_actions['pv'].groupby(valid_actions['user_id']).sum()) / \
#         (valid_actions['pv'].groupby(valid_actions['user_id']).count())
# merge.to_csv('./mergey.csv')

merge = pd.read_csv('./merge.csv')
mergey = pd.read_csv('./mergey.csv')
df = pd.concat([merge,train_ground['label']],axis=1)

features = ['pv']
X_train = df.loc[:, features].values
Y_train = df.loc[:, ['label']].values
X_test = mergey
x = StandardScaler().fit_transform(X_train)

# 随机梯度下降
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)