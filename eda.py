# %%

#Handle table-like data and matrices
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pickle


# Modelling algorithms
import xgboost as xgb
from sklearn.linear_model import LinearRegression

# Modelling helpers
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


pal = sns.color_palette()
# %matplotlib inline
pd.options.display.max_colwidth = 200

# os.chdir("./Reezocar")
# os.getcwd()

print("File size :")
for f in os.listdir('./input'):
     print(f + '   ' + str(round(os.path.getsize('./input/' + f)/1000000, 2)) + 'MB')
# %% Import data

df_toyota = pd.read_csv('./input/toyota.csv')

# %% describe

df_toyota.head(30)
df_toyota.describe()
df_toyota.isnull().any()

# %%  Data preparation
df_train = df_toyota

global df_train_x

# delete outliers
df_train_x = df_train.drop(df_train[df_train.Price > 30000].index)

# Feature Engineering

fuel_type = pd.get_dummies(df_train_x.FuelType)
# print(fuel_type.head())
# df_train_x = df_train_x.drop(["FuelType"], axis=1)
df_train_x = df_train_x.drop(["FuelType", "CC", "MetColor", "Doors"], axis=1)

df_train_x = pd.concat([df_train_x, fuel_type], axis=1)
print(df_train_x.head())
print(df_train.describe())
# %% Function definition

def ecdf(arg):
    # Number of data points: n
    n = len(arg)
    # x-data for the ECDF: x
    x = np.sort(arg)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

def plot_correlation_map(df):
    corr = df.corr()
    _ , ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    # cmap = sns.cubehelix_palette(light = 1, as_cmap=True)
    _ = sns.heatmap(
    corr,
    cmap = cmap,
    square = True,
    cbar_kws = {'shrink' : 0.9},
    ax=ax,
    annot=True,
    annot_kws={'fontsize' : 12}
    )
    plt.show()

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    plt.show()

def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()
    plt.show()

def plot_variable_importance(x, y):
    tree = DecisionTreeClassifier(random_state = 99)
    tree.fit(x, y)
    plot_model_var_imp(tree, x, y)

def gen_dash():
    # Fig 1
    fig1 = plt.figure(0, figsize=(20,10))
    fig1.add_subplot(521)
    _ = plt.hist(df_train["Price"])
    _ = plt.xlabel('price')
    _ = plt.ylabel('Nb of cars')
    plt.tight_layout(pad=0.4, w_pad=6, h_pad=1.0)
    # Fig2
    fig1.add_subplot(522)
    x, y = ecdf(df_train["Price"])
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel('Price')
    _ = plt.ylabel('ECDF')
    #fig3
    fig1.add_subplot(523)
    _ = plt.hist(df_train["Age"])
    _ = plt.xlabel('Age (years)')
    _ = plt.ylabel('Nb of cars')
    #fig4
    fig1.add_subplot(524)
    _ = plt.hist(df_train["KM"])
    _ = plt.xlabel('Kilometers')
    _ = plt.ylabel('Nb of cars')
    #fig5
    fig1.add_subplot(525)
    _ = plt.hist(df_train["HP"])
    _ = plt.xlabel('HP')
    _ = plt.ylabel('Nb of cars')
    #fig6
    fig1.add_subplot(526)
    _ = plt.hist(df_train["CC"])
    _ = plt.xlabel('CC')
    _ = plt.ylabel('Nb of cars')
    #fig7
    fig1.add_subplot(527)
    _ = plt.hist(df_train["Weight"])
    _ = plt.xlabel('Weight')
    _ = plt.ylabel('Nb of cars')


    plot_categories(df_train, cat='Price', target='FuelType')
    plot_distribution(df_train, var='Price', target='MetColor')
    plot_distribution(df_train, var='Price', target='Automatic')
    plot_distribution(df_toyota, var='Price', target='Doors')

    plot_correlation_map(df_train)

def model_training(df_train_x):

    train_valid_y = df_train_x.Price
    train_valid_x = df_train_x.drop(["Price"], axis=1)
    # print(train_valid_x.ix[12])
    train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size=.7)

    model = LinearRegression()
    model.fit(train_x, train_y)

    # params = {}
    # params['objective'] = 'reg:linear'
    # params['eval_metric'] = 'logloss'
    # params['eta'] = 0.02
    # params['max_depth'] = 4
    #
    # d_train = xgb.DMatrix(train_x, label=train_y)
    # d_valid = xgb.DMatrix(valid_x, label=valid_y)
    #
    # watchlist = [(d_train, 'd_train'), (d_valid, 'd_valid')]
    #
    # bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)



    print("Linear Model Scores :\ntraining data %2f \nvalidation data: %2f"%(model.score(train_x, train_y), model.score(valid_x, valid_y)))

    return model

def main():
    # gen_dash()

    model = model_training(df_train_x)
    filename = "optimized_linear.sav"
    joblib.dump(model, filename)
    # model = joblib.load(filename)

    df_test_x =  df_train_x.drop(["Price"], axis=1)

    # d_test = xgb.DMatrix(df_test_x.ix[1])
    # p_test = model_xgb.predict(d_test)

    print(df_train_x.ix[1])
    print("Linear prediction : ", model.predict(df_test_x.ix[1]))
    # print("XGBoost prediciton : ", p_test)

# %% Launch main
if __name__ == '__main__':
    main()
