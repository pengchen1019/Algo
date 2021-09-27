import pandas as pd
import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 数据压缩
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
            start_mem - end_mem) / start_mem))
    return df


# 处理空值
def process_null(data, a):
    for dataset in data:
        mean = train_data[a].mean()
        std = test_data[a].std()
        is_null = dataset[a].isnull().sum()
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        verify_slice = dataset[a].copy()
        verify_slice[np.isnan(verify_slice)] = rand_age
        dataset[a] = verify_slice
    return dataset


# 数据导入
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

# 相关性热力图绘制
# data_corr = train_captcha.corr()
# plt.subplots(figsize=(9, 9), dpi=100, facecolor='w')
# fig = sns.heatmap(data_corr, annot=True, vmax=1, square=True, cmap="Blues", fmt='.2g')
# fig.set_title('Train_captcha相关性热力图')
# plt.show()

train_pv = pd.read_csv('./train_pv.csv')
train_verify = pd.read_csv('./train_verify.csv')
train_server = pd.read_csv('./train_server.csv')

test_pv = pd.read_csv('./test_pv.csv')
test_verify = pd.read_csv('./test_verify.csv')
test_server = pd.read_csv('./test_server.csv')

#数据融合
train_data = pd.merge(train_pv, train_verify, on='user_id', how='left')
train_data = pd.merge(train_data, train_server, on='user_id', how='left')
train_data = pd.merge(train_data, train_ground, on='user_id', how='left')
train_data.to_csv('./train_data.csv', index=False)

test_data = pd.merge(test_pv, test_verify, on='user_id', how='left')
test_data = pd.merge(test_data, test_server, on='user_id', how='left')
test_data = pd.merge(test_data, valid_set, on='user_id', how='left')
test_data.to_csv('./test_data.csv', index=False)

# 融合后的数据
train_data = pd.read_csv('./train_data.csv')
test_data = pd.read_csv('./test_data.csv')

# 空值处理
data = [train_data, test_data]
process_null(data, 'verify_time')
process_null(data, 'server_time')

# 数据压缩
train_data = reduce_mem_usage(train_data)
test_data = reduce_mem_usage(test_data)

# 训练数据生成
features = ['pv', 'verify_time', 'server_time']
X_train = train_data.loc[:, features].values
Y_train = train_data.loc[:, ['label']].values
F_test = test_data.loc[:, features].values

# 随机森林
random_forest = RandomForestClassifier(criterion="gini",
                                       min_samples_leaf=10,
                                       max_depth=9,
                                       min_samples_split=80,
                                       n_estimators=70,
                                       oob_score=True,
                                       random_state=10,
                                       n_jobs=-1
                                       )
random_forest.fit(X_train, Y_train.ravel())
Y_prediction = random_forest.predict(F_test)
random_forest.score(X_train, Y_train)
print("oob score:", round(random_forest.oob_score_, 4) * 100, "%")

#结果输出
Y_prediction = pd.DataFrame(Y_prediction)
result = pd.concat([valid_set['user_id'], Y_prediction], axis=1)
result.columns = ['user_id', 'label']
result.to_csv('./result.csv', index=None)
