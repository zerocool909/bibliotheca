# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 08:48:51 2019

@author: adraj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:30:52 2019

@author: adraj
"""

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")

customers = pd.read_csv('user_master_bib.csv') 
transactions = pd.read_csv('book_transactions_per_user.csv')
book_purchase_history = pd.read_csv('BOOKSPURCHHISTORY.csv',delimiter = ',',encoding='latin1')
user_transaction = book_purchase_history[['UserID','BookID']]
all_user_purchase =  pd.merge(customers,book_purchase_history, on='UserID',how='left')
trainbooks =  pd.read_csv('book_master_train_bib.csv') 
testbooks =  pd.read_csv('book_master_test_bib.csv') 
all_books =  pd.read_csv('book_master_all_books_bib.csv')
testbook_id_only = pd.read_csv('book_test_matrix.csv')


del all_user_purchase['AGEGROUP']
del all_user_purchase['SUBSTATE']
del all_user_purchase['WeekofYear']
del all_user_purchase['TIMESTAMP']

all_user_purchase['BookID'] = all_user_purchase['BookID'].fillna(0)



transactions.columns
n = transactions['UserID'].unique()
# example 2: organize a given table into a dataframe with customerId, single productId, and purchase count
s=time.time()

data = pd.melt(all_user_purchase.set_index('UserID')['BookID'].apply(pd.Series).reset_index(), 
             id_vars=['UserID'],
             value_name='BookID') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['UserID', 'BookID']) \
    .agg({'BookID': 'count'}) \
    .rename(columns={'BookID': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'BookID': 'BookID'})
data['BookID'] = data['BookID'].astype(np.int64)

sub=pd.read_csv('data_booksBought.csv')

data = sub
print("Execution time:", round((time.time()-s)/60,2), "minutes")


print (data.head())
print (transactions.head())

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy

data_dummy = create_data_dummy(data)
#normalize item values across users
df_matrix = pd.pivot_table(data, values='purchase_count', index='UserID', columns='BookID')
df_matrix.head()
#total unique customers in transcations dataset
"""customers_num = data['UserID'].unique()
df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
print(df_matrix_norm.shape)
df_matrix_norm.head()

# create a table for input to the modeling
d = df_matrix_norm.reset_index()
d.index.names = ['scaled_purchase_freq']
print(d.index.names)
data_norm = pd.melt(d, id_vars=['UserID'], value_name='scaled_purchase_freq').dropna()
print(data_norm.shape)
data_norm.head()
"""
"""df= experiment        
bins = [0, 1, 5, 10, 25, 50, 100]
labels = [1,2,3,4,5,'higher group']
df['binned'] = pd.cut(df['customerId'], bins=bins, labels=labels)   """

"""def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()    """ 
       
train, test = train_test_split(data, test_size = .2)
print(train.shape, test.shape)

train_data = tc.SFrame(train)
test_data = tc.SFrame(test)
customers_sc = tc.SFrame(customers)
trainbooks_sc = tc.SFrame(trainbooks)
testbooks_sc = tc.SFrame(testbooks)
all_books_sc = tc.SFrame(all_books)
testbook_id_only_sc = tc.SFrame(testbook_id_only)



# We can define a function for this step as follows
def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# lets try with both dummy table and scaled/normalized purchase table
train_data_dummy, test_data_dummy = split_data(data_dummy)
#train_data_norm, test_data_norm = split_data(data_norm) 

#model starts
user_id = 'UserID'
item_id = 'BookID'
target = 'purchase_count'
users_to_recommend = list(transactions[user_id])
n_rec = 10 # number of items to recommend
n_display = 30

popularity_model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target,
                                                    item_data=all_books_sc,
                                                    user_data=customers_sc)


popularity_recomm = popularity_model.recommend(users=users_to_recommend, k=n_rec)
popularity_recomm.print_rows(n_display)

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target,
                                                    item_data=all_books_sc,
                                                    user_data=customers_sc
                                                    )
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target,
                                                    item_data=all_books_sc,
                                                    user_data=customers_sc,                                                    
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target,
                                                    item_data=all_books_sc,
                                                    user_data=customers_sc,
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model


# variables to define field names
# constant variables include:
user_id = 'UserID'
item_id = 'BookID'
users_to_recommend = list(customers[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to print the head / first few rows in a defined dataset
#Using purchase dummy
# these variables will change accordingly
name = 'popularity'
target = 'purchase_count'
pop = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#popularity bases
name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

"""name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)"""

#cosine based Purchase count
name = 'cosine'
target = 'purchase_count'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#cosine based purchase dummy
name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

"""#cosine based ON NORMALIZED DF
name = 'cosine'
target = 'scaled_purchase_freq'
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)"""
#PEARSON OVER NORMALIZED DATA\
#PEARSON based Purchase count
name = 'pearson'
target = 'purchase_count'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

#PEARSON based purchase dummy
name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

"""name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)"""

#EVALUATION TIME
# create initial callable variables
models_w_counts = [pop, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
#models_w_norm = [pop_norm, cos_norm, pear_norm]

names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
#names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts', 'Pearson Similarity on Scaled Purchase Counts']

#model on count
eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
#model on purchase dummy
eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)
#model on scaled purchase counts
#eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)

#post selecting best model based on RMSE,PRECISION AND RECALL
users_to_recommend = list(customers[user_id])

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='purchase_count',
                                            item_data=all_books_sc, 
                                            user_data=customers_sc,
                                            similarity_type='cosine')

m2 = tc.ranking_factorization_recommender.create(tc.SFrame(data_dummy),
                                                 user_id=user_id, 
                                                 item_id=item_id,
                                                 target='purchase_count',
                                                 user_data=customers_sc,
                                                 item_data=all_books_sc)


recom = final_model.recommend(users=users_to_recommend,items=testbook_id_only_sc, k=n_rec)
recom.print_rows(n_display)

recom = m2.recommend(users=users_to_recommend,items=testbook_id_only_sc, k=n_rec)
recom.print_rows(n_display)


#Creating  output function
def create_output(model, users_to_recommend, n_rec, print_csv=True):
    #recomendation = final_model.recommend(users=users_to_recommend, k=n_rec,items=testbook_id_only_sc)
    recomendation = model.recommend(users=users_to_recommend, k=n_rec,items=testbook_id_only_sc)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['UserID', 'recommendedProducts']].drop_duplicates() \
        .sort_values('UserID').set_index('UserID')
    if print_csv:
        df_output.to_csv('option2_recommendation_2003_2.csv')
        print("An output file can be found in 'output' folder with name 'option1_recommendation_2003.csv'")
    return df_output

df_output = create_output(m2, users_to_recommend, n_rec, print_csv=True)
print(df_output.shape)
df_output.head()

#customer recommendation function
def customer_recomendation(customer_id):
    if customer_id not in df_output.index:
        print('Customer not found.')
        return customer_id
    return df_output.loc[customer_id]

print(customer_recomendation(600008))
