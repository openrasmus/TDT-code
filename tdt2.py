import pandas as pd
import numpy as np
import xgboost as xgb

#Define variables
ID_COLUMN = 'Id'
TARGET_COLUMN = 'Response'
seed = 0
CHUNKSIZE = 50000
NROWS = 250000

#File paths, change to wherever you have them saved 
train_numeric_path = "train_numeric.csv/train_numeric.csv"
train_date_path = "train_date.csv/train_date.csv"
test_numeric_path = "test_numeric.csv/test_numeric.csv"
test_date_path = "test_date.csv/test_date.csv"

train = pd.read_csv(train_numeric_path, usecols=[ID_COLUMN, TARGET_COLUMN], nrows=NROWS)
test = pd.read_csv(test_numeric_path, usecols=[ID_COLUMN], nrows=NROWS)

#Data cleaning
# train["StartTime"] = -1
# test["StartTime"] = -1

# nrows = 0
# for tr, te in zip(pd.read_csv(train_date_path, chunksize=CHUNKSIZE), pd.read_csv(test_date_path, chunksize=CHUNKSIZE)):
#     feats = np.setdiff1d(tr.columns, [ID_COLUMN])
#     stime_tr = tr[feats].min(axis=1).values
#     stime_te = te[feats].min(axis=1).values
#     train.loc[train.Id.isin(tr.Id), 'StartTime'] = stime_tr
#     test.loc[test.Id.isin(te.Id), 'StartTime'] = stime_te
#     nrows += CHUNKSIZE
#     if nrows >= NROWS:
#         break

# #Combine train and test, since we use cross validation. fillna to fix NANs
# ntrain = train.shape[0]
# train_test = pd.concat((train, test)).reset_index(drop=True).reset_index(drop=False)
# train_test['1'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
# train_test['2'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
# train_test = train_test.sort_values(by=['StartTime', 'Id'], ascending=True)
# train_test['3'] = train_test[ID_COLUMN].diff().fillna(9999999).astype(int)
# train_test['4'] = train_test[ID_COLUMN].iloc[::-1].diff().fillna(9999999).astype(int)
# train_test = train_test.sort_values(by=['index']).drop(['index'], axis=1)

# train = train_test.iloc[:ntrain, :]
# features = np.setdiff1d(list(train.columns), [TARGET_COLUMN, ID_COLUMN])
# y = train.Response.ravel()
# train = np.array(train[features])
# prior = np.sum(y) / (1.*len(y))
# dtrain=xgb.DMatrix(train, label=y)

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 2,
    'eval_metric': 'auc',
    'base_score': prior
}


param = {
    "verbosity":0,
    'objective': 'binary:logistic',
    'seed': 0,
    'eval_metric': 'auc',
    'base_score': prior,
    'colsample_bytree': 0.7,#Between (0,1]
    'subsample': 0.7,      #Between 0-1
    'learning_rate': 0.1,  #Between 0-1
    'max_depth': 4,        #Between 0-infty
    'num_parallel_tree': 1,#Number of trees(forrest)
    'min_child_weight': 2
}



def find_best_parameters():
  etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
  subsamples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
  colsamples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
  num_trees = [1,2,3,4,5]
  max_depths = range(1,10)
  max_acc = 0.8883
  best_param = {}  
  for eta in etas:
    param["learning_rate"]=eta  
    for subsample in subsamples: 
      param["subsample"] = subsample  
      for colsample in colsamples:
          param["colsample_bytree"] = colsample
          for num_tree in num_trees:
              param["num_parallel_tree"] = num_tree
              for max_depth in max_depths:
                  param["max_depth"] = max_depth
                  mean,std = model(param)
                  if mean>max_acc:
                      print(param)
                      max_acc = mean
                      best_param = param
  return(best_param,max_acc)


def model(param):
    rf = xgb.cv(param, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True,
             early_stopping_rounds=3, verbose_eval=0, show_stdv=True)
    mean = rf.iloc[-1,2]
    std = rf.iloc[-1,1]
    return(mean,std)


# print(find_best_parameters())

test= {'verbosity': 0, 'objective': 'binary:logistic', 'seed': 0, 'eval_metric': 'auc', 'base_score': 0.00564, 'colsample_bytree': 0.7, 'subsample': 0.7, 'learning_rate': 0.1, 'max_depth': 9, 'num_parallel_tree': 5, 'min_child_weight': 2}
print(model(test))
#Trains XGBoost algorithm
# extreme = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True,
              # early_stopping_rounds=1, verbose_eval=1, show_stdv=True)
 
#Trains random forrest, when num_boost_round=1 XGB is same as RF

#Print results
# rf_mean = rf.iloc[-1, 0]
# rf_std = rf.iloc[-1, 1]

# cv_mean = extreme.iloc[-1, 0]
# cv_std = extreme.iloc[-1, 1]

# print('CV-Mean-XGB: {0}+{1}'.format(cv_mean, cv_std))
# print('CV-Mean-RF: {0}+{1}'.format(rf_mean, rf_std))
