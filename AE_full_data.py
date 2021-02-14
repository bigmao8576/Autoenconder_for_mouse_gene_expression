# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import utils
import numpy as np
import os
import argparse

import tensorflow as tf
from RNN_AE import rnn_ae
import sys





def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    
    parser.add_argument('--c_th', type=float,
                        default=0.005,
                        help='the cost threshold for stopping training, default = 0.005')
  
    parser.add_argument('--hidden_num', type=int,
                        default=32,
                        help='the dimension of hidden state, default = 40')
    
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
    
    parser.add_argument('--dataset', type=str,
                        default='0601',
                        help='dataset type, we have 0601 and 1030, but 1030 is currently not available, default = 0601')
    
    
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
    int_ep=args.int_ep
    
    gpu_num = args.gpu
    
    data_name = args.dataset
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
    
    plotdata = {
                "train_cost":[],
                "train_recover_data":[],
                "train_representation":[],
                } 
    
    plotdata['config'] = {
            'int_ep':int_ep,
            'cost_threshold':cost_th,
            'hidden_num':hidden_num,
            'lr':lr,
            #'thresh':thresh,
            'BATCH_SIZE':BATCH_SIZE,
            }
    
    # create the folders for saving the results
    first_folder = 'Alldata_results_%s_h%d_bs%d_lr%0.5f_l%0.5f'%(data_name,hidden_num,BATCH_SIZE,lr,cost_th)
    if not os.path.exists(first_folder):
        os.mkdir(first_folder)
    
    plotdata_save_path = os.path.join(first_folder,'all_results')
    if not os.path.exists(plotdata_save_path):
        os.mkdir(plotdata_save_path)
    
    
  
        
    checkpoint_path = os.path.join(plotdata_save_path,'check_point','cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
        
    
    
    
    # import data
    
    if data_name =='0601':
        csv_path =  'data_rhythm_20200601.csv'
        channel_num=2
    elif data_name =='1030':
        csv_path =  'dataProcessed.csv'
        channel_num=3
    
    data_dict,sample_len = utils.file2dict(csv_path,channel_num=channel_num,norm=True)
    x_train,name =  utils.data2array(data_dict)

    ch_num = x_train.shape[-1]
    assert ch_num == channel_num

    
    
    

      
    train_ds = tf.data.Dataset.from_tensor_slices((aug_data(x_train)))
    train_ds = train_ds.shuffle(3000)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(10)
    
    
    rnn_auto = rnn_ae(hidden_num,ch_num)
    optimizer = tf.keras.optimizers.Adam(lr)
    train_func = train_step_L(optimizer,rnn_auto)
    
    train_loss = eval_loss(x_train,rnn_auto)

    ep = 0
    
    print('Before training, the training loss is %0.8f'%(train_loss))
    
    
    
    while train_loss>cost_th:
        for x_data in train_ds:
            train_func(x_data)
    
        if ep%int_ep==0:  
            
            train_loss = eval_loss(x_train,rnn_auto)

            plotdata["train_cost"].append(train_loss)

            
          
            train_x_rec,train_la = rnn_auto(x_train)
            print('At epoch %d, the training loss is %0.8f '%(ep,train_loss))
            
         
            utils.draw_pic_cost(plotdata,plotdata_save_path)
            utils.draw_pic_sample(x_train,train_x_rec.numpy(),plotdata_save_path)
        
            pickle.dump(plotdata,open(os.path.join(plotdata_save_path,'plotdata.pkl'),'wb'),protocol=2)
            
            rnn_auto.save_weights(checkpoint_path)
            
        ep+=1
            
    train_x_rec,train_la = rnn_auto(x_train)
    
    
    plotdata["train_recover_data"]=train_x_rec.numpy()
    plotdata["train_representation"]=train_la.numpy()
    
    plotdata["train_data"]=x_train
    plotdata["train_name"]=name
    
    pickle.dump(plotdata,open(os.path.join(plotdata_save_path,'plotdata.pkl'),'wb'),protocol=2)
    utils.draw_pic_sample(x_train,train_x_rec.numpy(),plotdata_save_path)

if __name__=='__main__': 

    sys.exit(main(sys.argv[1:])) 





