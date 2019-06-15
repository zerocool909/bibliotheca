# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:57:08 2019

@author: adraj
"""

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

book_catalogue = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\BOOKSCATALOGUE.csv',delimiter = ',',encoding='latin1')
book_master_test = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\BOOKSMASTERTEST.csv',delimiter = ',',encoding='latin1')
book_master_train = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\BOOKSMASTERTRAIN.csv',delimiter = ',',encoding='latin1')
book_purchase_history = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\BOOKSPURCHHISTORY.csv',delimiter = ',',encoding='latin1')
book_visit_history = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\BOOKSVISITHISTORY.csv',delimiter = ',',encoding='latin1')
user_master = pd.read_csv('C:\\Users\\adraj\\Desktop\\MacchineLearningPython\\library_book_prediction\\initial datasets\\USERMASTER.csv',delimiter = ',',encoding='latin1')
import os
print(os.listdir("../db"))
read =  pd.read_csv('../db/user_master_bib.csv')
"""user_master2 = airport_freq.merge(airports[airports.ident == 'KLAX'][['id']], 
                                  left_on='airport_ref', right_on='id', how='inner')
                                    [['airport_ident', 'type', 'description', 'frequency_mhz']]"""






del user_master['SIGNUPDATE']
del user_master['GENDER']
user_master.dtypes
null_columns=user_master.columns[user_master.isnull().any()]
user_master[null_columns].isnull().sum()
#state can be one encoded using below calls
user_master['STATE'] = user_master['STATE'].fillna('no_state')
val = user_master['STATE'].unique()
dfDummies = pd.get_dummies(user_master['GENDER'], prefix = 'GENDER')
user_master = pd.concat([user_master, dfDummies], axis=1)
dfDummies_2 = pd.get_dummies(user_master['STATE'], prefix = 'STATE')
user_master = pd.concat([user_master, dfDummies], axis=1)


book_master_train.dtypes
book_master_train['BookID'] = book_master_train['BookID'].apply(lambda x: [int(i) for i in x ])
null_columns=book_master_train.columns[book_master_train.isnull().any()]
book_master_train[null_columns].isnull().sum()
book_master_train = book_master_train[['BookID','GENRE','AUTHOR','USERRATINGS','Popularity']]
dfDummies = pd.get_dummies(book_master_train['GENRE'], prefix = 'GENRE')
book_master_train = pd.concat([book_master_train, dfDummies], axis=1)
del book_master_train.columns['GENRE']

book_master_test.dtypes
book_master_test['BookID'] = book_master_test['BookID'].apply(lambda x: [int(i) for i in x ])
null_columns=book_master_test.columns[book_master_test.isnull().any()]
book_master_test[null_columns].isnull().sum()
book_master_test = book_master_test[['BookID','GENRE','AUTHOR','USERRATINGS','Popularity']]
dfDummies = pd.get_dummies(book_master_test['GENRE'], prefix = 'GENRE')
book_master_test = pd.concat([book_master_test, dfDummies], axis=1)
del book_master_test['GENRE']
book_master_test['GENRE_Horror'] =0
book_master_train.columns

all_user_data = pd.merge(user_master,book_purchase_history, on='UserID',how='left')
all_user_data_details = pd.merge(all_user_data,book_master_train, on='BookID',how='left')

all_user_data_subscription =  pd.merge(user_master,book_visit_history, on='UserID',how='left')
del all_user_data_subscription['BOOKPREVIEWED'] 
del all_user_data_subscription['SESSIONID']  
del all_user_data_subscription['REFERRALID']   
del all_user_data_subscription['TIMESTAMP']   
all_user_data_subscription_details = pd.merge(all_user_data_subscription,book_master_train, on='BookID',how='left')


df = all_user_data_details

df2 = df.groupby('UserID', as_index=False).agg(lambda x: x.tolist())

df3 = all_user_data_subscription_details  

df3 = df3.groupby('UserID', as_index=False).agg(lambda x: x.tolist())

df_pb = df2[['UserID','BookID','GENRE','Popularity']]
df_sb = df3[['UserID','BookID','GENRE','USERRATINGS']]

df_pb.rename(index=str, columns={"BookID": "PURCHASE_BOOK_ID", "GENRE": "PBGENRE"},inplace = True)
df_sb.rename(index=str, columns={"BookID": "SUBSCRIBED_REV_BOOK_ID", "GENRE": "SCGENRE"},inplace = True)

bookdata = pd.merge(df_pb,df_sb, on='UserID',how='left')

def uniques(xs):
    return list(set(xi for x in xs for xi in x))


bookdata['ALL_BOOK_ID'] = bookdata[['PURCHASE_BOOK_ID', 'SUBSCRIBED_REV_BOOK_ID']].apply(uniques, axis=1)
bookdata['ALL_BOOK_GENRE'] = bookdata[['PBGENRE', 'SCGENRE']].apply(uniques, axis=1)

bookdata['ALL_BOOK_GENRE'] = bookdata['ALL_BOOK_GENRE'].apply(lambda x: [i for i in x if str(i) != "nan"])
bookdata['ALL_BOOK_ID'] = bookdata['ALL_BOOK_ID'].apply(lambda x: [i for i in x if str(i) != "nan"])
bookdata['USERRATINGS'] = bookdata['USERRATINGS'].apply(lambda x: [i for i in x if str(i) != "nan"])

bookdata['ALL_BOOK_ID'] = bookdata['ALL_BOOK_ID'].apply(lambda x: [int(i) for i in x ])

bookdata['USERRATINGS'] = bookdata['USERRATINGS'].apply(lambda x: [int(i) for i in x ])
bookdata['USERRATINGS'] = bookdata['USERRATINGS'].apply(lambda x: [0 for i in x if str(i) == "nan"])

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(bookdata.ALL_BOOK_ID)
bookdata_matrix = bookdata.join(pd.DataFrame(X, columns=mlb.classes_))
"""
def Average(lst): 
    return sum(lst) / len(lst) 

bookdata['AVG_RATING'] = bookdata['USERRATINGS'].apply(Average('USERRATINGS'))"""



items_counts = book_purchase_history['BookID'].value_counts(sort=False)
top = items_counts.loc[[items_counts.idxmax()]]
value, count = top.index[0], top.iat[0]

bookdata_matrix.columns
cols = [col for col in bookdata_matrix.columns if col not in ['PURCHASE_BOOK_ID', 'Popularity','SCGENRE','ALL_BOOK_GENRE',
                                                              'USERRATINGS','SUBSCRIBED_REV_BOOK_ID','PBGENRE']]
matrix_df = bookdata_matrix[cols]

matrix_df['book_boughts'] =matrix_df['ALL_BOOK_ID'].str.len() 

print(matrix_df['book_boughts'] [1])

sub = book_master_test
sub.to_csv('book_master_test_bib.csv',index=False)



bookdata["all_bookid"] = df_pb["PURCHASE_BOOK_ID"].toList() + df_sb["SUBSCRIBED_REV_BOOK_ID"].toList()
bookdata["all_bookid"] = pd.concat([df_pb["PURCHASE_BOOK_ID"], df_sb["SUBSCRIBED_REV_BOOK_ID"]], axis=0).reset_index()
df3['AGEGROUP']
['GENDER']
['STATE']
['BOOKP']
bookdata.dtypes
pd.to_numeric(bookdata.PURCHASE_BOOK_ID, downcast='integer')

bookdata[['PURCHASE_BOOK_ID', 'three']] = df[['two', 'three']].astype(float)
bookdata = bookdata.infer_objects()

conv = bookdata[['PURCHASE_BOOK_ID']]
results = list(map(int, conv))


