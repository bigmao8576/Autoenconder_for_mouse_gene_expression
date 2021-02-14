#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:13:50 2020

@author: bigmao
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense,GRU,Bidirectional
from tensorflow.keras import Model


class rnn_ae(Model):
    def __init__(self,rnn_c, la_ch):
        super(rnn_ae, self).__init__()
    
        self.la_ch = la_ch
        self.rnn_c = rnn_c
                 
        self.en_forward = GRU(self.rnn_c,
                   return_sequences=False,
                   return_state=True,
                   recurrent_initializer='glorot_uniform')
        
        self.en_backward = GRU(self.rnn_c,
                   return_sequences=False,
                   return_state=True,
                   recurrent_initializer='glorot_uniform',
                   go_backwards=True)
        
        
        self.en_bd = Bidirectional(layer=self.en_forward,backward_layer=self.en_backward)
        
        
        self.trans_la = Dense(self.rnn_c)
        
        self.trans_ini = Dense(self.rnn_c*2)
        
        
        
        self.de_forward = GRU(self.rnn_c,
                           return_sequences=True,
                           return_state=False,
                           recurrent_initializer='glorot_uniform')   
    
        self.de_backward = GRU(self.rnn_c,
                           return_sequences=True,
                           return_state=False,
                           recurrent_initializer='glorot_uniform',
                           go_backwards=True)  
        
        
        self.de_bd = Bidirectional(layer=self.de_forward,backward_layer=self.de_backward) 
        
        
        self.output_layer = Dense(self.la_ch)
        

    def call(self, x):

        last_states,_,_= self.en_bd(inputs = x)
        
        la_final = self.trans_la(last_states)
        
        de_init = self.trans_ini(la_final)
        
        zes = tf.zeros([x.shape[0],x.shape[1],1])
               
        rnn_output = self.de_bd(inputs = zes,initial_state=[de_init[:,:self.rnn_c],de_init[:,self.rnn_c:]])
        
        output = self.output_layer(rnn_output)
        

        
        return output,la_final
    
    
    
    
    