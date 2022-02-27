#!/usr/bin/env python
# coding: utf-8

# ## Projet 2021-2022 / Yu Rui ; Nicolas Diaz

# #### Charger Dataset

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from boruta import BorutaPy


# In[4]:


df = pd.read_csv('dataset.csv', index_col = 'ID', sep=';')
df


# #### l'analyse exploratoire des données.

# In[6]:


df.columns


# In[7]:


df.shape


# In[8]:


df.dtypes.value_counts()


# In[9]:


for col in df.select_dtypes('object'):
    print(f'{col :=<20} {df[col].unique()}')


# Valeurs NAN Manquent

# In[10]:


plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)


# In[11]:


(df.isna().sum()/df.shape[0]).sort_values(ascending=True) ##Pourcentage des valeurs manquent


# #### Mesuere statistique

# Moyenne

# In[5]:


df_num = df.replace({'A' : 2, 'B' : 3, 'C' : 4, 'D' : 5, 'E' : 6, 'F' : 7, 'G' : 8, 'H' : 9})
df_num["VAL_M3"] = pd.to_numeric(df_num["VAL_M3"])
df1 = df_num.drop(columns = ['PAT_SEX'])
df_mean = np.mean(df1)
print("Moyenne:\n",df_mean)


# In[10]:


moy_gr = df_mean.plot.bar()
plt.xlabel('Type_Dent')
plt.ylabel('Moy')
plt.title('Moyenne',fontweight ="bold")


# L'ecart

# In[11]:


ecart_type = np.std(df1)
print("L'ecart-type:\n",ecart_type)


# In[12]:


ecart_grap = ecart_type.plot.bar()
plt.xlabel('Type_Dent')
plt.ylabel('Ecart')
plt.title('l’écart-type',fontweight ="bold")


# Histogrames des variables continues

# In[13]:


df['VAL_I1'].hist(label='VAL_I1')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_I1',fontweight ="bold")


# In[14]:


df['VAL_I2'].hist(label='VAL_I2')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_I2',fontweight ="bold")


# In[31]:


df['VAL_C1'].hist(label='VAL_C1')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_C1',fontweight ="bold")


# In[32]:


df['VAL_PM2'].hist(label='VAL_PM2')
#df2['VAL_PM2'].hist(bin = 25)
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_PM2',fontweight ="bold")


# In[33]:


df['VAL_M1'].hist(label='VAL_M1')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_M1',fontweight ="bold")


# In[34]:


df['VAL_M2'].hist(label='VAL_M2')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_M2',fontweight ="bold")


# In[35]:


df['VAL_M3'].hist(label='VAL_M3')
plt.xlabel('Stades Maturité')
plt.ylabel('Distribution')
plt.title('Type Dient: VAL_M3',fontweight ="bold")


# #### Nettoyage des données.

# In[6]:


df_num = df.replace({'A' : 2, 'B' : 3, 'C' : 4, 'D' : 5, 'E' : 6, 'F' : 7, 'G' : 8, 'H' : 9})
df_num["VAL_M3"] = pd.to_numeric(df_num["VAL_M3"])
# df.describe()
df_num


# In[4]:


df_num.describe()


# In[7]:


df['VAL_M3'].fillna(0)


# In[8]:


df_num["PAT_AGE"] = df_num["PAT_AGE"].apply(np.round) 
df_num['VAL_M3'].fillna(value = 0, inplace = True)
df_num


# In[9]:


df_mean = df_num.fillna(value = df_num.mean(), inplace = True)
for col in df.select_dtypes('object'):
    df_num[col] = df_num[col].round(decimals = 0)
print(df_num)


# In[10]:


df_num['PAT_AGE'].value_counts().sort_values(ascending=True)


# #### Preprocessing

# In[8]:


df_num['PAT_AGE'].value_counts(normalize = True)


# ### Données d'Entrenaiment et Test

# In[11]:


X = df_num.drop(columns='PAT_AGE')
y = df_num['PAT_AGE']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size = 0.20)


# ### Procédure d'évaluation

# In[12]:


def evaluation(model):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
#     print(ypred.shape)
    print("Score MAE: ",mean_absolute_error(y_test, ypred))   
    print("Score RMSE: ",np.sqrt(mean_squared_error(y_test, ypred)))   
#     print("Score MSE: ",mean_squared_error(y_test, y_pred))
#     print("Score Median AE: ",median_absolute_error(y_test, y_pred))
    
#     err_hist = np.abs(y - ypred)
#     plt.hist(err_hist, bins = 50)
#     plt.show


# In[30]:


mlp = MLPRegressor(random_state=0)
xgc = xgb.XGBRegressor(random_state=0)
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
tree = tree.DecisionTreeRegressor()
RDF = RandomForestRegressor(random_state=0)


# In[25]:


list_models = {   
                'XGBoost' : xgc,
                'MLP': mlp,
                'SVM': regr,
                'Tree': tree,
                'RandomForest': RDF
                }


# In[29]:


for name, model in list_models.items():
    print(name)
    evaluation(model)


# ### Optimisation des modeles

# SVM

# In[11]:


# Ensemble des paramètres et leurs valeurs
parametres = {
                'C': [1, 10], 
                'gamma': ('scale', 'auto'),
                'kernel': ('rbf', 'poly', 'sigmoid')
             }

# GridSearchCV
modele = SVR()
grille = GridSearchCV(modele, parametres, cv = 5, n_jobs = -1, verbose = 2)
grille.fit(X_train, y_train)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)


# Decision Tree

# In[12]:


# Ensemble des paramètres et leurs valeurs
parametres = {
              "criterion": ("mse", "mae"),
              "min_samples_split": (10, 20, 40),
              "max_depth": (10, 40, 100),
              "min_samples_leaf": (20, 40, 100),
              "max_leaf_nodes" : (10, 40, 100)
             }

# GridSearchCV
modele = tree.DecisionTreeRegressor()
grille = GridSearchCV(modele, parametres, cv = 5, n_jobs = -1, verbose = 2)
grille.fit(X_train, y_train)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)


# MLP

# In[13]:


# Ensemble des paramètres et leurs valeurs
parametres = {'hidden_layer_sizes':[80,100],
              'activation':('tanh','relu'),
              'solver':('lbfgs','sgd','adam'),
              'learning_rate':('constant','adaptive')
             }

# GridSearchCV
modele = MLPRegressor()
grille = GridSearchCV(modele, parametres, cv = 5, n_jobs = -1, verbose = 2)
grille.fit(X_train, y_train)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)


# XGBoost

# In[14]:


# Ensemble des paramètres et leurs valeurs
parametres = { 'max_depth': (2, 4, 6),
           'n_estimators': (100, 500, 1000),
           'colsample_bytree': (0.2, 0.6, 0.8),
           'min_child_weight': (3, 5, 7),
           'gamma': (0, 1, 5),
           'subsample': (0.4, 0.6, 0.8)
              }

# GridSearchCV
modele = xgb.XGBRegressor()
grille = GridSearchCV(modele, parametres, cv = 5, n_jobs = -1, verbose = 2)
grille.fit(X_train, y_train)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)


# Random Rorest

# In[20]:


# Ensemble des paramètres et leurs valeurs
# parametres = { 
#     'n_estimators': (100, 150, 200, 250, 300),
#     'max_depth': (1,2,3,4)
#               }

parametres = { 
    'n_estimators': [100, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [1,2,3,4]
}

# GridSearchCV
modele = RandomForestRegressor(random_state=0)
grille = GridSearchCV(modele, parametres, cv = 5, n_jobs = -1, verbose = 2)
grille.fit(X_train, y_train)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)


# ## L'utilisation de Boruta pour selectioner les mieux features

# Une fois que le modèle optimal pour l'entreiment et la prédiction a été identifié. Nous déterminons les meilleures caractéristiques pour optimiser davantage le modèle.

# In[26]:


trans = BorutaPy(RDF, random_state=42, verbose=2)
sel = trans.fit_transform(X.values, y.values)


# In[27]:


trans.support_


# In[28]:


X.columns[trans.support_]


# In[31]:


# Transformer le dataset pour supprimer les features non importantes
X_trans  = trans.transform(X.values)


# In[40]:


X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_trans, y,  test_size = 0.20)


# In[41]:


grille.fit(X_train_b, y_train_b)

# Meilleur score
print("Meilleur score :", grille.best_score_)
print("Meilleur jue de paramètres :", grille.best_params_)

