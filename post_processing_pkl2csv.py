#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script used to convert the results in pickle files to CSV. 

"""

import pickle
import pandas as pd
import os
import numpy as np
import utils
# two sets of data. the datasets 0601 is currently available, 1030 will be released in recent future 
data_name = '0601'
data_name = '1030'


if data_name == '0601':  
    fold_folder_name = 'fold_results_0601_h32_bs512_lr0.00100_l0.00100'
    fulldata_folder_name = 'Alldata_results_0601_h32_bs512_lr0.00100_l0.00200'
    original_table = 'data_rhythm_20200601.csv'
    channel_num=2

else:
    fold_folder_name = 'fold_results_1030_h16_bs512_lr0.00100_l0.00100'
    fulldata_folder_name = 'Alldata_results_1030_h16_bs512_lr0.00100_l0.00100'
    original_table = 'dataProcessed.csv'
    channel_num=3

# get the column names
df = pd.read_csv(original_table, dtype={'Name':str, 'Value':float})
head_names = list(df.columns)
# first let deal with the 5-fold results
fold_dict = pickle.load(open('fold_dict_%s.pkl'%data_name,'rb'), encoding='latin1')

if not os.path.exists(fold_folder_name):
    raise FileNotFoundError('the file %s dosenot exist'%fold_folder_name)
    
fold_name = os. listdir(fold_folder_name)

for fold_num in fold_name:
    sub_folder = os.path.join(fold_folder_name,fold_num)
    plotdata_path = os.path.join(sub_folder,'plotdata.pkl')
    plotdata = pickle.load(open(plotdata_path,'rb'), encoding='latin1')
    
    ori_train_data = fold_dict[fold_num]['train_data']
    ori_test_data = fold_dict[fold_num]['test_data']
    
    rec_train_data = plotdata['train_recover_data']
    rec_test_data = plotdata['test_recover_data']
    
    
    train_name = fold_dict[fold_num]['train_name']
    test_name = fold_dict[fold_num]['test_name']
    
    ori_train_data = np.reshape(ori_train_data,[ori_train_data.shape[0],ori_train_data.shape[1]*ori_train_data.shape[2]])
    ori_test_data = np.reshape(ori_test_data,[ori_test_data.shape[0],ori_test_data.shape[1]*ori_test_data.shape[2]])

    rec_train_data = np.reshape(rec_train_data,[rec_train_data.shape[0],rec_train_data.shape[1]*rec_train_data.shape[2]])
    rec_test_data = np.reshape(rec_test_data,[rec_test_data.shape[0],rec_test_data.shape[1]*rec_test_data.shape[2]])

    df = pd.DataFrame(ori_train_data)
    df.insert(loc=0, column='0', value=train_name)
    df.columns = head_names
    save_path = os.path.join(sub_folder,'original_train_data.csv')
    df.to_csv(save_path, index=False)
    
    df = pd.DataFrame(ori_test_data)
    df.insert(loc=0, column='0', value=test_name)
    df.columns = head_names
    save_path = os.path.join(sub_folder,'original_test_data.csv')
    df.to_csv(save_path, index=False)
    
    df = pd.DataFrame(rec_train_data)
    df.insert(loc=0, column='0', value=train_name)
    df.columns = head_names
    save_path = os.path.join(sub_folder,'rebuilt_train_data.csv')
    df.to_csv(save_path, index=False)
    
    df = pd.DataFrame(rec_test_data)
    df.insert(loc=0, column='0', value=test_name)
    df.columns = head_names
    save_path = os.path.join(sub_folder,'rebuilt_test_data.csv')
    df.to_csv(save_path, index=False)
    
    latent_train = plotdata['train_representation']
    latent_test = plotdata['test_representation']
    
    df = pd.DataFrame(latent_train)
    df.insert(loc=0, column='0', value=train_name)
    save_path = os.path.join(sub_folder,'train_latent_vector.csv')
    df.to_csv(save_path, index=False,header=False)
    
    df = pd.DataFrame(latent_test)
    df.insert(loc=0, column='0', value=test_name)
    save_path = os.path.join(sub_folder,'test_latent_vector.csv')
    df.to_csv(save_path, index=False,header=False)
    
# now let's deal with the full dataset
if not os.path.exists(fulldata_folder_name):
    raise FileNotFoundError('the file %s dosenot exist'%fulldata_folder_name) 
sub_folder = os.path.join(fulldata_folder_name,'all_results')
plotdata_path = os.path.join(sub_folder,'plotdata.pkl')

# load the resulting file
plotdata = pickle.load(open(plotdata_path,'rb'), encoding='latin1')

ori_train_data = plotdata['train_data']
rec_train_data = plotdata['train_recover_data']
train_name = plotdata['train_name']
latent_train = plotdata['train_representation']


ori_train_data = np.reshape(ori_train_data,[ori_train_data.shape[0],ori_train_data.shape[1]*ori_train_data.shape[2]])
rec_train_data = np.reshape(rec_train_data,[rec_train_data.shape[0],rec_train_data.shape[1]*rec_train_data.shape[2]])


df = pd.DataFrame(ori_train_data)
df.insert(loc=0, column='0', value=train_name)
df.columns = head_names
save_path = os.path.join(sub_folder,'original_train_data.csv')
df.to_csv(save_path, index=False)

df = pd.DataFrame(rec_train_data)
df.insert(loc=0, column='0', value=train_name)
df.columns = head_names
save_path = os.path.join(sub_folder,'rebuilt_train_data.csv')
df.to_csv(save_path, index=False)

df = pd.DataFrame(latent_train)
df.insert(loc=0, column='0', value=train_name)
save_path = os.path.join(sub_folder,'train_latent_vector.csv')
df.to_csv(save_path, index=False,header=False)














