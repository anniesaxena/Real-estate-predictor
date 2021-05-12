#importing necessary libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

#The Dataset

pd.set_option('display.max_rows', None)
df1 = pd.read_csv("dataset/Bengaluru_House_Data.csv")
df1.head()

# Step 1 : Data Cleaning

df1.groupby('area_type')['area_type'].agg('count')
#droping unnecessary columns

df2 = df1.drop(['area_type','availability','society','balcony'], axis='columns')
df2.head()
#checking the null values

df2.isna().sum()
#droping null rows

df3 = df2.dropna()
df3.isna().sum()
#inconsistency in data

df3['size'].unique()
df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3['BHK'].unique()
#inconsistency in data

df3['total_sqft'].unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)].head()

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) ==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()

df4.loc[30]

# Step 2 : Feature Engineering

df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 /df5['total_sqft']
df5.head()

len(df5['location'].unique())
#high dimensionality problem

df5.location = df5.location.apply(lambda x : x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats

len(location_stats[location_stats <=10])

location_stats_less_than_10 = location_stats[location_stats <=10]
location_stats_less_than_10

len(df5.location.unique())

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())

df5.head(10)

# Step 3 : Outlier removal 

#examining dataset by using sqft-per-bedroom threshold

df5[df5.total_sqft/df5.BHK <300].head()

df6 = df5[~(df5.total_sqft/df5.BHK <300)]
df6.shape

#checking extreme cases in price_per_sqft

df6.price_per_sqft.describe()

#filtering out data which is beyond one standard deviation
#doing this per location because price_per_sqft depends on location too

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape

#examining cases where price of 3BHK is less then price of 2BHK with same total_sqft

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color='green',marker ='+', label='3 BHK', s=50)
    plt.xlabel('Total sqft area')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()

plot_scatter_chart(df7,'Hebbal')

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'st' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)
df8.shape

plot_scatter_chart(df8,'Hebbal')

#dataset has normal distribution now

plt.hist(df8.price_per_sqft, rwidth=0.8)
plt.xlabel("Price_per_sqft")
plt.ylabel("Count")

df8.bath.unique()

#removing bathrooms anomalies

df8[df8.bath>10]

plt.hist(df8.bath, rwidth=0.8)
plt.xlabel("NO. of Bathrooms")
plt.ylabel("Count")

#outliers

df8[df8.bath>df8.BHK+2]

df9 =  df8[df8.bath<df8.BHK+2]
df9.shape

#removing unnecessary columns

df10 = df9.drop(['size','price_per_sqft'], axis='columns')
df10.head()

#   Step 4: Model Building

#converting categorical information to numerical information by using one hot encoding

dummies = pd.get_dummies(df10.location)
dummies.head()

df11 = pd.concat([df10,dummies.drop('other', axis='columns')], axis='columns')
df11.head()

df12 = df11.drop('location', axis='columns')
df12.head()

df12.shape

#X is all independent variables

X = df12.drop('price', axis='columns')
X.head()

y = df12.price
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

# using k fold cross validation

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X,y, cv=cv)

#using grid search cv : for figuring out best score by running on various regressors and parameters

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchCV(X,y):
    algos = {
        'linear regression': {
            'model' : LinearRegression(),
            'params' : {
                'normalize' : [True, False]
            }
        },
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
    }
    
    scores =[]
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'],config['params'],cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model','best_score','best_params'])


find_best_model_using_gridsearchCV(X,y)

#using linear regression for building price predictor

def predict_price(location,sqft,bath,bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >=0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]


predict_price('1st Phase JP Nagar',1000, 2, 2)

predict_price('1st Phase JP Nagar',1000, 3, 3)

# exporting model to pickle file

import pickle
with open("banglore_home_prices_model.pickle",'wb') as f:
    pickle.dump(lr_clf,f)

#for columns data

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json",'w') as f:
    f.write(json.dumps(columns))
