
from keras.models import load_model
from joblib import load
import xgboost as xgb
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker



plt.rcParams['font.sans-serif'] = [ 'SimHei']  

plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(r"C:\Users\Dave\Desktop\工艺参数论文\材料成分(没有工艺).xlsx", usecols=range(10), nrows=(80))  # using you data path . data hvae not process
data.columns = data.columns.astype(str) 
data_estimaore = pd.read_excel(r"C:\Users\Dave\Desktop\工艺参数论文\工艺参数-材料成分.xlsx", usecols=range(14), nrows=(80))
data_estimaore.columns = data_estimaore.columns.astype(str) 
X = data_estimaore.iloc[:, 4:]
y = data_estimaore.iloc[:, 3] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#save model
with open('pipeline_浇注温度-80.pkl', 'rb') as f:    # save your model
    # upload model
    estimator1 = pickle.load(f)
with open('pipeline_挤压压力.pkl', 'rb') as f:
  
    estimator2 = pickle.load(f)
with open('pipeline_保压时间.pkl', 'rb') as f:
  
    estimator3 = pickle.load(f)
with open('pipeline_模具预热温度.pkl', 'rb') as f:
  

    #
    estimator6 = pickle.load(f)
#creat explainer
explainer_shap_pipeline_gy1 = shap.KernelExplainer(estimator1.predict, X_train, keep_index=True)
explainer_shap_pipeline_gy2 = shap.KernelExplainer(estimator2.predict, X_train, keep_index=True)
explainer_shap_pipeline_gy3 = shap.KernelExplainer(estimator3.predict, X_train, keep_index=True)
explainer_shap_pipeline_gy4 = shap.KernelExplainer(estimator4.predict, X_train, keep_index=True)
explainer_shap_pipeline_gy5 = shap.KernelExplainer(estimator5.predict, X_train1, keep_index=True)
explainer_shap_pipeline_gy6 = shap.KernelExplainer(estimator6.predict, X_train1, keep_index=True)





shap_values_pipeline_gy1 = explainer_shap_pipeline_gy1.shap_values(data)  #save your SHAP values

with open('shap_values_pipeline_gy1.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy1, f)
shap_values_pipeline_gy2 = explainer_shap_pipeline_gy2.shap_values(data)

with open('shap_values_pipeline_gy2.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy2, f)
shap_values_pipeline_gy3 = explainer_shap_pipeline_gy3.shap_values(data)

with open('shap_values_pipeline_gy3.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy3, f)
shap_values_pipeline_gy4 = explainer_shap_pipeline_gy4.shap_values(data)

with open('shap_values_pipeline_gy4.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy4, f)


shap_values_pipeline_gy1_Explanation = explainer_shap_pipeline_gy1(data)
shap_values_pipeline_gy2_Explanation = explainer_shap_pipeline_gy2(data)
shap_values_pipeline_gy3_Explanation = explainer_shap_pipeline_gy3(data)
shap_values_pipeline_gy4_Explanation = explainer_shap_pipeline_gy4(data)

#save explanation shap values
with open('shap_values_pipeline_gy1_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy1_Explanation, f)
with open('shap_values_pipeline_gy2_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy2_Explanation, f)
with open('shap_values_pipeline_gy3_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy3_Explanation, f)
with open('shap_values_pipeline_gy4_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy4_Explanation, f)
with open('shap_values_pipeline_gy5_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy5_Explanation, f)
with open('shap_values_pipeline_gy6_Explanation.pkl', 'wb') as f:
    pickle.dump(shap_values_pipeline_gy6_Explanation, f)

#upload shap values

with open('shap_values_pipeline_gy1.pkl', 'rb') as f:
    shap_values_pipeline_gy1 = pickle.load(f)

with open('shap_values_pipeline_gy2.pkl', 'rb') as f:
    shap_values_pipeline_gy2 = pickle.load(f)

with open('shap_values_pipeline_gy3.pkl', 'rb') as f:
    shap_values_pipeline_gy3 = pickle.load(f)

with open('shap_values_pipeline_gy4.pkl', 'rb') as f:
    shap_values_pipeline_gy4 = pickle.load(f)

#upload explanation shap values

with open('shap_values_pipeline_gy1_Explanation.pkl', 'rb') as f:
    shap_values_pipeline_gy1_Explanation = pickle.load(f)
with open('shap_values_pipeline_gy2_Explanation.pkl', 'rb') as f:
    shap_values_pipeline_gy2_Explanation = pickle.load(f)
with open('shap_values_pipeline_gy3_Explanation.pkl', 'rb') as f:
    shap_values_pipeline_gy3_Explanation = pickle.load(f)
with open('shap_values_pipeline_gy4_Explanation.pkl', 'rb') as f:
    shap_values_pipeline_gy4_Explanation = pickle.load(f)


shap_values_pipeline_gy = [shap_values_pipeline_gy1, shap_values_pipeline_gy2, shap_values_pipeline_gy3, shap_values_pipeline_gy4]

#creat summary plot -pt
shap.summary_plot(shap_values_pipeline_gy1, data, feature_names=data.columns, show=False) 
plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\浇注温度-70-summaryPlot图.png')
plt.clf()
#creat summary plot sp
shap.summary_plot(shap_values_pipeline_gy2, data, feature_names=data.columns, show=False)  
plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\挤压压力-70-summaryPlot图.png')
plt.clf()
#creat summary plot dp
shap.summary_plot(shap_values_pipeline_gy3, data, feature_names=data.columns, show=False)  
plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间-70-summaryPlot图.png')
plt.clf()
#creat summary plot dt
shap.summary_plot(shap_values_pipeline_gy4, data, feature_names=data.columns, show=False)  
plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\模具预热温度-70-summaryPlot图.png')
plt.clf()


#creat force plot dt
force_plot_visualizer_gy1 = shap.force_plot(explainer_shap_pipeline_gy1.expected_value, shap_values_pipeline_gy1, data, feature_names=data.columns, show=False)
shap.save_html('C:\\Users\\Dave\\Desktop\\SHAP图\\浇注温度-全部数据点的解释的力图.html', force_plot_visualizer_gy1) 
#creat force plot sp
force_plot_visualizer_gy2 = shap.force_plot(explainer_shap_pipeline_gy2.expected_value, shap_values_pipeline_gy2, data, feature_names=data.columns, show=False)
shap.save_html('C:\\Users\\Dave\\Desktop\\SHAP图\\挤压压力-全部数据点的解释的力图.html', force_plot_visualizer_gy2)
#creat force plot dp
force_plot_visualizer_gy3 = shap.force_plot(explainer_shap_pipeline_gy3.expected_value, shap_values_pipeline_gy3, data, feature_names=data.columns, show=False)
shap.save_html('C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间-全部数据点的解释的力图.html', force_plot_visualizer_gy3)
#creat force plot dt
force_plot_visualizer_gy4 = shap.force_plot(explainer_shap_pipeline_gy4.expected_value, shap_values_pipeline_gy4, data, feature_names=data.columns, show=False)
shap.save_html('C:\\Users\\Dave\\Desktop\\SHAP图\\模具预热温度-全部数据点的解释的力图.html', force_plot_visualizer_gy4)





#creart bar plot ,which were collection all feature's absolute mean SHAP value .
for i in range(6):
    if i < 4:
        shap.summary_plot(shap_values_pipeline_gy[i], data, feature_names=data1.columns, plot_type='bar',
                            show=False)
    elif i >=4:
        shap.summary_plot(shap_values_pipeline_gy[i], data1, feature_names=data1.columns, plot_type='bar',
                      show=False)
    ax = plt.gca()
    for p in ax.patches:
        width = p.get_width()
        ax.text(x=width,
                y=p.get_y() + (p.get_height() / 2),
                s='{:.5f}'.format(width),
                va='center')
    if i == 0:
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\浇注温度-summaryPlot重要性图.png')
      plt.clf()
    if i == 1:
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\挤压压力-summaryPlot重要性图.png')
      plt.clf()
    if i == 2 :
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间-summaryPlot重要性图.png')
      plt.clf()
    if i == 3:
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\模具预热温度-summaryPlot重要性图.png')
      plt.clf()
    if i == 4:
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\抗拉强度-summaryPlot重要性图.png')
      plt.clf()
    if i == 5:
      plt.savefig('C:\\Users\\Dave\\Desktop\\SHAP图\\屈服强度-summaryPlot重要性图.png')
      plt.clf()
#creat a figure for all features SHAP values
fig = plt.figure(figsize=(20,10))

ax0 = fig.add_subplot(141)
ax0.title.set_text('浇注温度')
shap.plots.bar(shap_values_pipeline_gy1_Explanation, show=False)
# shap.summary_plot(shap_values_pipeline_gy1, data, feature_names=data.columns, plot_type='bar',show = False) 
ax0.set_xlabel(r'SHAP values',fontsize = 11)
plt.subplots_adjust(wspace=2)

ax1 = fig.add_subplot(142)
ax1.title.set_text('挤压压力')
shap.plots.bar(shap_values_pipeline_gy2_Explanation, show=False)
ax1.set_xlabel(r'SHAP values',fontsize = 11)
plt.subplots_adjust(wspace=2)

ax2 = fig.add_subplot(143)
ax2.title.set_text('保压时间')
shap.plots.bar(shap_values_pipeline_gy3_Explanation, show=False)
ax2.set_xlabel(r'SHAP values',fontsize = 11)
ax2.set_xticks([0, 5])
plt.subplots_adjust(wspace=2)

ax3 = fig.add_subplot(144)
ax3.title.set_text('模具预热温度')
shap.plots.bar(shap_values_pipeline_gy4_Explanation, show=False)
ax3.set_xlabel(r'SHAP values',fontsize = 11)
plt.subplots_adjust(wspace=2)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\重要度排名(工艺参数).png')
plt.close()
#
#creat independence plot
features = data.columns
for features in features:
    shap.dependence_plot(features, shap_values_pipeline_gy3, data, show=False)
    plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\保压时间\\PDP依赖图\\{features}-PDP特征依赖图.png')
    plt.clf()

features = data.columns
# print(features)
# print(shap_values_pipeline_gy3)
for features in features:
    shap.dependence_plot(features, shap_values_pipeline_gy3, data, interaction_index=None, show=False)
    plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间\\单特征依赖图\\单特征-{features}特征依赖图.png')
    plt.clf()

#create shap.heatmap
shap.plots.heatmap(shap_values_pipeline_gy1_Explanation, show=False)  #pt
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\浇注温度\\热图.png')
plt.close()
shap.plots.heatmap(shap_values_pipeline_gy2_Explanation, show=False)  #sp
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\挤压压力\\热图.png')
plt.close()
shap.plots.heatmap(shap_values_pipeline_gy3_Explanation, show=False)  #dp
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间\\热图.png')
plt.close()
shap.plots.heatmap(shap_values_pipeline_gy4_Explanation, show=False)  #dt
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\模具预热温度\\热图.png')
plt.close()

#shap violet
name = list(X_train.columns)
shap.plots.violin(shap_values_pipeline_gy1, features=X_train, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\浇注温度\\提琴图.png')
plt.close()
shap.plots.violin(shap_values_pipeline_gy2, features=X_train, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\挤压压力\\提琴图.png')
plt.close()
shap.plots.violin(shap_values_pipeline_gy3, features=X_train, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\保压时间\\提琴图.png')
plt.close()
shap.plots.violin(shap_values_pipeline_gy4, features=X_train, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\模具预热温度\\提琴图.png')
plt.close()
shap.plots.violin(shap_values_pipeline_gy5, features=X_train1, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\抗拉强度\\提琴图.png')
plt.close()
shap.plots.violin(shap_values_pipeline_gy6, features=X_train1, feature_names=name, plot_type="layered_violin"
, show=False)
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\屈服强度\\提琴图.png')
plt.close()

# creat waterfall plot
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
shap.plots.waterfall(shap_values_pipeline_gy1_Explanation[0], show=False) # For the first observation
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\局部解释-瀑布图\\浇注温度-样本1.png')
plt.close()
shap.plots.waterfall(shap_values_pipeline_gy2_Explanation[0], show=False) # For the first observation
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\局部解释-瀑布图\\挤压压力-样本1.png')
plt.close()
shap.plots.waterfall(shap_values_pipeline_gy3_Explanation[0], show=False) # For the first observation
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\局部解释-瀑布图\\保压时间-样本1.png')
plt.close()
shap.plots.waterfall(shap_values_pipeline_gy4_Explanation[0], show=False) # For the first observation
plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\局部解释-瀑布图\\模具预热温度-样本1.png')
plt.close()

# creat independence plot
features = data.columns
for i in range(4):
    if i == 0:
        shap_values_pipeline_gy = shap_values_pipeline_gy1
        file = '浇注温度'
    elif i == 1:
        shap_values_pipeline_gy = shap_values_pipeline_gy2
        file = '挤压压力'
    elif i == 2:
        shap_values_pipeline_gy = shap_values_pipeline_gy3
        file = '保压时间'
    elif i == 3:
        shap_values_pipeline_gy = shap_values_pipeline_gy4
        file = '模具预热温度'
    for ii in features:
        shap.dependence_plot(ii, shap_values_pipeline_gy, data, alpha=0.36,show = False)
        plt.savefig(f'C:\\Users\\Dave\\Desktop\\SHAP图\\部分依赖图\\{file}\\单特征-{ii}特征依赖图.png')
        plt.clf()






