'''

1. Create a model through a pipeline
2. The pipeline includes data cleaning, data partitioning, model partitioning weights, integrated models, and then outputs prediction results
3. Output information about the integrated model, etc
4. Continuously finding models that comply with MAE MRE through a for loop is also a method.
5. Prepare for automation process
'''
from hypergbm import make_experiment
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
import logging
import pickle
import random
from hypergbm import HyperGBM
from hypergbm.search_space import search_space_general
from hypergbm.tests import test_output_dir
from hypernets.core.callbacks import SummaryCallback, FileLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers.random_searcher import RandomSearcher
from hypernets.tabular.datasets import dsutils
import shap
from hypernets.core.random_state import set_random_state
import numpy as np
from hypernets.searchers import RandomSearcher, EvolutionSearcher
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
from matplotlib.ticker import MaxNLocator
from hypergbm.search_space import GeneralSearchSpaceGenerator, Real
from hypernets.core.search_space import Choice, Int
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from hypergbm.search_space import GeneralSearchSpaceGenerator
from hypernets.core.search_space import Real, Choice, Int
from hypergbm.cfg import HyperGBMCfg as cfg
from hypernets.core import randint

from hypernets.tabular.datasets import dsutils
from hypergbm import make_experiment
from tqdm import tqdm

from IPython.display import HTML
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
start_time = time.time()

data = pd.read_excel(r"D:\工艺参数论文\工艺参数-材料成分0.01-铸型尺寸.xlsx",   #import data
                     usecols=range(14))
data.columns = data.columns.astype(str)  


# split train data and test data
X = data.iloc[:, 4:]
y = data.iloc[:, 0]  #import


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
# squeeze all data
train_data = pd.concat([X_train, y_train], axis=1)

data_cleaner_args = {
    'correct_object_dtype': True,
    'drop_columns': None,
    'drop_constant_columns': False,
    'drop_duplicated_columns':False,
    'drop_idness_columns': True,
    'drop_label_nan_rows': True,
    'int_convert_to': 'float', 'nan_chars': None,
    'reduce_mem_usage': False,
    'reserve_columns': None
}
# define your hyper parameters range
my_space1=GeneralSearchSpaceGenerator(
#choose model
enable_xgb=True,
enable_lightgbm=True,
enable_catboost=True,
enable_histgb=True,

xgb_init_kwargs = {
'booster': 'dart',
'max_depth':Int(3, 25),
'n_estimators': Choice([100,150,200,250,300]),
'learning_rate': Real(0.0001,1),
'min_child_weight': Real(0.001,1),
'gamma': Real(0.0001,1),
'reg_alpha': Real(0.0001,1),
'reg_lambda': Real(0.0001,1),
'colsample_bytree':Real(0.0001,1),
'subsample':Real(0.0001,1),
},
lightgbm_init_kwargs={
'boosting_type': 'gbdt',
'num_leaves': Int(10, 200),
'max_depth': Int(3, 10),
'learning_rate': Real(0.001, 0.3),
'n_estimators': Choice([50, 100, 150, 200, 250, 300]),
'subsample_for_bin': Int(20000, 300000),
'min_split_gain': Real(0.0, 1.0),
'min_child_weight': Real(0.001, 0.1),
'subsample': Real(0.5, 1.0),
'colsample_bytree': Real(0.5, 1.0),
'reg_alpha': Real(0.0, 1.0),
'reg_lambda': Real(0.0, 1.0),
},
catboost_init_kwargs = {
'depth': Int(4, 10),
'learning_rate': Real(0.01, 0.3),
'l2_leaf_reg': Real(1, 10),
'border_count': Int(32, 255),
'thread_count': Int(1, 7),
'bagging_temperature': Real(0, 1),
'random_strength': Real(0, 1),
},
histgb_init_kwargs = {
'max_iter': Int(100, 500),
'max_depth': Int(3, 25),
'learning_rate': Real(0.001, 1),
'min_samples_leaf': Int(20, 200),
'l2_regularization': Real(0.0, 1.0),
'max_bins': Int(200, 255),
'max_leaf_nodes': Int(31, 128),
'scoring': 'loss',
'tol': Real(1e-7, 1e-1),
},
)
random_seeds = []
all_MRE = pd.DataFrame()
all_mae = pd.DataFrame()
all_r2 = pd.DataFrame()
time1 = pd.DataFrame()
for i in tqdm(range(200), desc="Processing"):
    # find best model
    # searcher = ['evolution', 'mcts', 'random','moead', 'nsga2', 'rnsga2']   #hyper paremeters searcher
    experiment = make_experiment(train_data, target=data.columns[0], reward_metric='rmse',
                                 data_cleaner_args=data_cleaner_args, searcher='random', search_space=my_space1)
    estimator = experiment.run()
    # save model
    with open(f'pipeline_.pkl', 'wb') as f:  #                             
        pickle.dump(estimator, f)
    # output model
    print("Best model:", estimator)
    # print("Best model parameters:", estimator.get_params()) #output model's hyper parameters
    # get the pred data
    pd.set_option('display.max_rows', None)
    y_pred = estimator.predict(X_test)
    #save data
    with open(f'测试集-输入.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open(f'测试集-输出.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open(f'预测值.pkl', 'wb') as f:
        pickle.dump(y_pred, f)
    # caculate mae
    mae = mean_absolute_error(y_test, y_pred)
    # caculate r2
    r2 = r2_score(y_test, y_pred)
    # caculate rmse
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    # get base model
    base_estimators = estimator.steps[-1][-1].estimators
    # output each base model information
    best_model = None
    for i, base_estimator in enumerate(base_estimators):
        if base_estimator is not None:  # 忽略权重为0的基模型
            model_params = base_estimator.model.get_params()
            # non_none_params = {k: v for k, v in model_params.items() }
            base_model_ = base_estimator.model
            print(f"基模型{i}的参数:", model_params)
            print(f"基模型{i}:", base_model_)
  
    MRE = np.mean(np.abs((y_test.values - y_pred) / y_test.values))
    print("Test set MAE:", mae)
    print("Test set R^2:", r2)
    print("Test set RMSE:", rmse)
    print("Test set MRE:", MRE)
    # caculate metrics 
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'MRE': np.mean(np.abs((y_test.values - y_pred) / y_test.values)),
        'r2': r2_score(y_test, y_pred),
        'rmse': sqrt(mean_squared_error(y_test, y_pred)),
    }
    # save model and metrics
    with open(f'_pipeline_metrics.pkl', 'wb') as f:                                     
        pickle.dump((estimator, metrics), f)
    all_mae = pd.concat([all_mae, pd.Series([mae])])
    all_MRE = pd.concat([all_MRE, pd.Series([MRE])])
    all_r2 = pd.concat([all_r2, pd.Series([r2])])
    print('MAE列表')
    print(all_mae)
    print('MRE列表')
    print(all_MRE)
    print('R2列表')
    print(all_r2)
    if mae <= 1 or r2 >= 0.2:            
        break
    end_time = time.time()
    elapsed_time = end_time - start_time
    time1 = pd.concat([time1, pd.Series([elapsed_time])])
    # print(f"程序运行时间: {time1}秒")
print('全部MAE列表')
print(all_mae)
print('全部MRE列表')
print(all_MRE)
print('全部R2列表')
print(all_r2)
print('全部运行时间列表')
print(time)