# -*- coding: utf-8 -*-
"""
@author: Shitong Mao
University of Pittsburgh 
bigmao_8576@hotmail.com

This code is used for k-fold cross-validation for AE
"""
import pickle
import utils
import numpy as np
import os
from copy import deepcopy
import argparse

import tensorflow as tf
from RNN_AE import rnn_ae
import sys





def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('fold_num', type=int,
                        help='fold number, must be 1-5')
    
    parser.add_argument('--c_th', type=float,
                        default=0.005,
                        help='the cost threshold for stopping training, default = 0.005')
  
    parser.add_argument('--hidden_num', type=int,
                        default=24,
                        help='the dimension of hidden state, default = 40')
    
    parser.add_argument('--thresh', type=int,
                        default=1,
                        help='threshold for gradient clipping, default = 1')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='batch size, default = 512')
    parser.add_argument('--int_ep', type=int,
                        default=1,
                        help='interpolation epoch, determining how many epochs for recording the loss. default = 10')

    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='the learning rate, default = 0.001')
    parser.add_argument('--gpu', type=int,
                        default=0,
                        help='specify the gpu number to train the model, default = 0')
    args = parser.parse_args(argv)
    return args 



def eval_loss(x,model):

    x_rec,_ = model(x)
    temp_loss = tf.reduce_mean((x_rec-x)**2)
        
    return temp_loss.numpy()



def aug_data(x):
    x = np.concatenate([x,np.flip(x,[1])])
    x = np.concatenate([x,np.flip(x,[2])])
    x = np.concatenate([x,x*(-1)])
    return x

def train_step_L(optimizer,model):

    @tf.function
    def apply_grads_L(x):
        
        with tf.GradientTape() as tape:
            x_rec,x_la = model(x)
            diff_loss = tf.reduce_mean(tf.keras.losses.MSE(x,x_rec))
        
        grad= tape.gradient(diff_loss, model.trainable_variables)
                
        grad,_ = tf.clip_by_global_norm(grad,1.0)
        
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
                  
    return apply_grads_L




def main(argv): 
    args = _parse_args(argv)
    
    cost_th= args.c_th
    hidden_num = args.hidden_num
    lr = args.lr
    BATCH_SIZE = args.batch_size
    fold_name = args.fold_num
    int_ep=args.int_ep
    
    gpu_num = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
    
    plotdata = {"epoch":[],
                "train_cost":[],
                "test_cost":[],
                "train_recover_data":[],
                "test_recover_data":[],
                "train_representation":[],
                "test_representation":[]
                } 
    
    plotdata['config'] = {
            'int_ep':int_ep,
            'cost_threshold':cost_th,
            'hidden_num':hidden_num,
            'lr':lr,
            #'thresh':thresh,
            'BATCH_SIZE':BATCH_SIZE,
            'fold_name':fold_name
            }
    
    # create the folders for saving the results
    first_folder = 'fold_results_0601_h%d_bs%d_lr%0.5f_l%0.5f'%(hidden_num,BATCH_SIZE,lr,cost_th)
    if not os.path.exists(first_folder):
        os.mkdir(first_folder)
    
    plotdata_save_path = os.path.join(first_folder,'fold_%d'%fold_name)
    if not os.path.exists(plotdata_save_path):
        os.mkdir(plotdata_save_path)
    
    
  
        
    checkpoint_path = os.path.join(plotdata_save_path,'check_point','cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
        
    # import data
    fold_dict = pickle.load(open('fold_dict_0601.pkl','rb'), encoding='latin1')
    fold_data = fold_dict['fold_%d'%fold_name]
    
    x_train=fold_data['train_data']
    x_test=fold_data['test_data']
    
    ch_num = x_train.shape[-1]
    #data_len,max_len = x_train.shape[:2]
    
    
    

      
    train_ds = tf.data.Dataset.from_tensor_slices((aug_data(x_train)))
    train_ds = train_ds.shuffle(3000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(10)
    
    
    rnn_auto = rnn_ae(hidden_num,ch_num)
    optimizer = tf.keras.optimizers.Adam(lr)
    train_func = train_step_L(optimizer,rnn_auto)
    
    train_loss = eval_loss(x_train,rnn_auto)
    test_loss = eval_loss(x_test,rnn_auto)
    ep = 0
    
    print('At fold %d the training and testing are %0.8f and %0.8f'%(fold_name, train_loss, test_loss))
    
    
    
    while train_loss>cost_th:
        for x_data in train_ds:
            train_func(x_data)
    
        if ep%int_ep==0:  
            
            train_loss = eval_loss(x_train,rnn_auto)
            test_loss = eval_loss(x_test,rnn_auto)
            plotdata["train_cost"].append(train_loss)
            plotdata["test_cost"].append(test_loss)
            
          
            train_x_rec,train_la = rnn_auto(x_train)
            print('At epoch %d, fold %d, the training and testing are %0.8f and %0.8f'%(ep,fold_name, train_loss, test_loss))
            
         
            utils.draw_pic_cost(plotdata,plotdata_save_path)
            utils.draw_pic_sample(x_train,train_x_rec.numpy(),plotdata_save_path)
        
            pickle.dump(plotdata,open(os.path.join(plotdata_save_path,'plotdata.pkl'),'wb'),protocol=2)
            
            rnn_auto.save_weights(checkpoint_path)
            
        ep+=1
            
    train_x_rec,train_la = rnn_auto(x_train)
    test_x_rec,test_la = rnn_auto(x_test)
    
    
    plotdata["train_recover_data"]=deepcopy(train_x_rec.numpy())
    plotdata["test_recover_data"]=deepcopy(test_x_rec.numpy())
    plotdata["train_representation"]=deepcopy(train_la.numpy())
    plotdata["test_representation"]=deepcopy(test_la.numpy())
    utils.draw_pic_sample(x_train,train_x_rec.numpy(),plotdata_save_path)

if __name__=='__main__': 

    sys.exit(main(sys.argv[1:])) 





