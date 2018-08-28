"=====================================第一部分：引入库============================================="
import pandas as pd
pd.set_option('display.width', 600) # 设置字符显示宽度
pd.set_option('display.max_rows', None) # 设置显示最大行
pd.set_option('display.max_columns', None) # 设置显示最大列，None为显示所有列

import numpy as np
np.set_printoptions(threshold=np.nan)
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls


import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold, StratifiedKFold

"=============================================第二部分：数据工程============================================="
'2.1 读取数据'
train = pd.read_csv('F:\Kaggle\Titanic\Dataset/train.csv')
test = pd.read_csv('F:\Kaggle\Titanic\Dataset/test.csv')
full_data = [train, test]
PassengerId = test['PassengerId']
# print(train.info(verbose=True)) # 训练集样本数为891，乘客年龄缺失187个，船舱缺失687，登船地点缺失2个
# print(test.info(verbose=True)) # 测试集样本数为418，乘客年龄缺失86个，票价缺失1个，船舱缺失327个
# print(type(train), type(PassengerId))
# print(train.head(3))
# print(PassengerId.head(3))
'2.2 Feature Engineering'
'Pclass'
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
# 显然，舱位等级越高，生存几率越大

'Sex'
print(train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean())
# 女性显然生存概率要远大于男性

'SibSp and Parch（being alone or not/family size）'
print(train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean())
# 观察数据可知，SibSp数目为3,4,5,8的人数应该是极少的（其数据都是‘整数’），其实作为判据并不公正
# 拥有一到两个SibSp（同龄伙伴）的人生存几率会比孤身一人有较明显的提高
print(train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean())
# 拥有父母或孩子会显著提高生存概率，大概可以理解为孩子们更容易得救，而为了使他们得到父母的照顾，父母中至少有一人会得救
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']+1
print(train[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean())
# 可以明显看出在1-4内，乘客的生存几率随着家庭规模的大小有显著的提高，而后出现急剧下降，推测是数据量太少引起的，不具备说服性
for dataset in full_data:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] ==1, 'isAlone'] = 1
print(train[['isAlone', 'Survived']].groupby('isAlone', as_index=False).mean())
# 明显看出孤身乘客生存几率要比非孤身乘客小得多

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
'Fare'
# 票价为0的一般为缺失值，通过下面一段代码得知票价数据缺失的样本数为15
# 样本索引为179,263,271,277,302,413,466,481,597,633,674,732,806,815,822
# j = 0
# for i in train[['Fare']].index:
#     if train[['Fare']].values[i] == 0:
#         j += 1
#         print(i, end=',')
# print('\n',j)
# 第一想法是检查这些样本所在的船舱，给予所在船舱的票价的中位数
# 通过观察发现，其实三等舱也有比较贵的票价，可能与头衔等其他因素有关，但是太过繁琐
# miss_indice = [179,263,271,277,302,413,466,481,597,633,674,732,806,815,822]
# samples_with_fare = train[['Fare', 'Pclass']].drop(miss_indice).groupby('Pclass', as_index=False)
# print(samples_with_fare.median()) # 1,2,3等舱位票价的中位数分别为61.9792, 15.0229和8.05
# print(samples_with_fare.max())
# print(samples_with_fare.mean())
# print(samples_with_fare.min())
for dataset in full_data:
    # dataset.loc[(dataset['Fare'] is np.nan) & (dataset['Pclass'] == 1), 'Fare'] = 61
    # dataset.loc[(dataset['Fare'] is np.nan) & (dataset['Pclass'] == 2), 'Fare'] = 15
    # dataset.loc[(dataset['Fare'] is np.nan) & (dataset['Pclass'] == 3), 'Fare'] = 8
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median(),)
# for i in miss_indice:
#     if train['Pclass'].values[i] == 1:
#         train['Fare'].values[i] = 61.9792
#     elif train['Pclass'].values[i] == 2:
#         train['Fare'].values[i] = 15.0229
#     elif train['Pclass'].values[i] == 3:
#         train['Fare'].values[i] = 8.05
# for i in test[['Fare']].index:
#     if test[['Fare']].values[i] == 0:
#         if test['Pclass'].values[i] == 1:
#             test['Fare'].values[i] = 61.9792
#         elif test['Pclass'].values[i] == 2:
#             test['Fare'].values[i] = 15.0229
#         elif test['Pclass'].values[i] == 3:
#             test['Fare'].values[i] = 8.05

# print(train[['Fare']].min())
# print(train[['Fare', 'Survived']].groupby('Survived', as_index=False).mean())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
# print(test[['Fare', 'Pclass']].groupby('Fare', as_index=False).mean())

'Embarked'
print(train.groupby(['Embarked'], as_index=False).count()) # 登船地点有两个数据缺失，先观察各个地点登船的人数
# 在S处有644人登船，Q处有77人，C处有168人
# 在这里原教程用最多的S来填补空缺
# 为了严谨我认为应该观察登船地点和票价、舱位等级、家庭大小等的关系，然后推断两个空缺处的登船信息
print(train[['Embarked', 'Pclass', 'Fare', 'FamilySize']].groupby('Embarked', as_index=False).mean())
# 观察发现，在C处上船的人阶层明显较高，票价最高，S处其次，Q处上船的都是劳苦大众
for i in train[['Embarked']].index:
    if not train[['Embarked']].values[i] == train[['Embarked']].values[i]:
        print(i, end=',')
# 这里注意np.nan是一种非常特殊的数值，判断一个值是否为nan，方法是判断其是否等于自身
# 61号缺失，住头等舱，标定其在C处上船；829也是
for i in [61,829]:
    if not train[['Embarked']].values[i] == train[['Embarked']].values[i]:
        train[['Embarked']] = train[['Embarked']].fillna('C')



'Age'
# 在训练集中有187个样本无年龄信息，测试集中86个
# 教程中生成（mean-std, mean+std)区间内的随机数，并且将年龄每16岁一组分成5组
# 我认为，年龄与票价、Parch有着极强的相关性，对Survived的预测极为重要，因此不能如此草率地生成随机数，但是我懒啊
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

print(train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
# print(train[['Age', 'Survived']].groupby('Age', as_index=False).mean())

'Name'
# 从人物的名字中我们可以得到他们的头衔，从而判断这些头衔与生存率的关系
def get_tittle(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_tittle)
print(pd.crosstab(train['Title'], train['Survived']))
# Col=Colnel，Major少校，为军衔
# Countess女伯爵，Don阁下，先生（多用于黑手党），Dona葡萄牙语女士，，Jonkheer一种荷兰贵族头衔，Sir爵士，Rev牧师
# Mlle法语小姐，Mme=Madame
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Dr', 'Jonkheer', 'Major', 'Lady', 'Rev', 'Sir'], 'BigPotato')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms', 'Dona'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Don', 'Mr')
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

'2.3 数据清洗与映射'
for dataset in full_data:
    # Sex
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1})
    # Titles
    title_map = {'BigPotato':1, 'Master':2, 'Miss':3, 'Mrs':4, 'Mr':5}
    # dataset[''] = dataset[].map()
    dataset['Title'] = dataset['Title'].map(title_map)
    dataset['Title'] = dataset['Title'].fillna(0)
    # Embarked
    embarked_map = {'S':0, 'Q':1, 'C':2}
    dataset['Embarked'] = dataset['Embarked'].map(embarked_map).astype(int)
    # Fare
    dataset.loc[dataset['Fare']<=7.925, 'Fare'] = int(0)
    dataset.loc[(dataset['Fare']>7.925)&(dataset['Fare'] <= 14.5), 'Fare'] = int(1)
    dataset.loc[(dataset['Fare']>14.5)&(dataset['Fare'] <=31.275), 'Fare'] = int(2)
    dataset.loc[dataset['Fare']>31.275, 'Fare'] = int(3)
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']= 4
# print(train[['Fare', 'Survived']].groupby('Survived', as_index=False).mean())
# Feature Selection
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize', 'CategoricalFare', 'CategoricalAge']
train = train.drop(drop_elements, axis=1)
test = test.drop(drop_elements[0:-2], axis=1)
# print(train.head(3), test.head(3), sep='\n')
# print(train.info(), test.info())

"=====================================第三部分：进行分类============================================="
'3.1 第一层分类器'
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)

class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    def predict(self, x):
        return self.clf.predict(x)
    def fit(self, x, y):
        return self.clf.fit(x, y)
    def feature_importance(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)

# def get_oof(clf, x_train, y_train, x_test):
#     oof_train = np.zeros((ntrain, )) # 保存训练集结果
#     oof_test = np.zeros((ntest, )) # 保存测试集最终结果
#     oof_test_skf = np.empty((NFOLDS, ntest)) # 保存NFOLDS次训练每次的测试集结果
#     for i, (train_index, val_index) in enumerate(kf):
#         x_train_i = x_train[train_index]
#         y_train_i = y_train[train_index]
#         x_val = x_train[val_index]
#
#         clf.train(x_train_i, y_train_i)
#
#         oof_train[val_index] = clf.predict(x_val)
#         oof_test_skf[i, :] = clf.predict(x_test)
#     oof_test[:] = oof_test_skf.mean(axis=0)
# # 假设数组a形状为（m, n），那么a.mean(axis=0)，就是取m个数的平均数，得到n个结果
# #  a.mean(axis=1)，就是取n个数的平均数，得到m个结果
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
rf_params = { 'n_jobs': -1,
    'n_estimators': 500,      # 该森林拥有五百棵树
     'warm_start': True,      # 设为TRUE的时候，利用前面的结果来为森林增加更多树
    'max_depth': 6,           # 树的最大深度
    'min_samples_leaf': 2,    # 每个叶节点处最少有多少个样本
    'max_features' : 'sqrt',  # 每个分割处最多考虑多少个特征，设为sqrt时max_features=sqrt(n_features)
    'verbose': 0}
#
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0}

ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75}

gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0}

svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
}
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# 将原始数据集转化为可以输入学习器的数组
y_train = train['Survived'].ravel()  # 将Survived数据摊平（类似flatten，但是flatten返回拷贝，ravel返回原数组）
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc, x_train, y_train, x_test) # Support Vector Classifier
print('Training is complete.')
def train_accuracy(oof_train, y_train):
    if not len(oof_train) == len(y_train):
        print('Length Error.')
        return 15
    right = 0 ; false = 0
    for i in range(len(oof_train)):
        if oof_train[i] == y_train[i]:
            right += 1
        else:
            false += 1
    return right/(right+false)
# print( 'Extra Trees:',train_accuracy(et_oof_train, y_train) )
# print( 'RandomForest:',train_accuracy(rf_oof_train, y_train) )
# print( 'AdaBoost:',train_accuracy(ada_oof_train, y_train) )
# print( 'Gradient Boost:',train_accuracy(gb_oof_train, y_train) )
# print( 'Support Vector Classifier:',train_accuracy(svc_oof_train, y_train) )
# tstclf = SVC(C=0.025, kernel='linear')
# tstclf.fit(x_train, y_train)
# print(train_accuracy(tstclf.predict(x_train), y_train))

'===================================3.2 Feature importances ===================================='
# To be continued


'===================================3.3 第二层分类模型-XGBoost ================================== '
x_train_2 = np.concatenate((et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test_2 = np.concatenate((et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
y_train_2 = y_train

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
# predictions = gbm.predict(x_test)
print(train_accuracy(gbm.predict(x_train), y_train_2))