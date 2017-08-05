#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2017/6/30 15:59
# @Author  : HouJP
# @Email   : houjp1992@gmail.com


from keras import backend
import tensorflow as tf


def binary_crossentropy_sum(y_true, y_pred):
    return backend.sum(backend.binary_crossentropy(y_pred, y_true), axis=-1)

from keras import backend as K

def matrix_pair_hinge_with_top(y_true, y_pred, class_num=1999, top_k = 30):
    
    y_true = K.tf.cast(y_true, tf.float32)
    
    y_pred_topk, y_pred_topk_idx = K.tf.nn.top_k(y_pred, k=top_k)
    batch_idx = K.tf.stack([K.tf.range(K.tf.shape(y_true)[0])]*top_k, axis=1)
    y_true_topk_idx = K.tf.stack([batch_idx, y_pred_topk_idx], axis=2)
    y_true_topk = K.tf.gather_nd(params=y_true, indices=y_true_topk_idx)

    s1 = K.tf.stack([y_pred_topk]*top_k, axis=1)
    s2 = K.tf.stack([y_pred_topk]*top_k, axis=2)
    g1 = K.tf.stack([y_true_topk]*top_k, axis=1)
    g2 = K.tf.stack([y_true_topk]*top_k, axis=2)
 
    s1 -= s2
    g1 -= g2

    hinge = (1.0 - s1 * g1) * K.tf.abs(g1)
    hinge = K.tf.maximum(0.0, hinge)

    loss = K.sum(K.sum(hinge, axis=1), axis=1)

    return loss

def matrix_pair_hinge_with_2side_top(class_num=1999, top_k_pred = 200, top_k_true = 20):
    
    def loss_func(y_true, y_pred):
        top_k = top_k_pred + top_k_true
        
        y_true = K.tf.cast(y_true, tf.float32)
        y_true_rnd = y_true + tf.random_uniform(tf.shape(y_true), 0, 0.2)
        
        y_pred_topk_t, y_pred_topk_idx = K.tf.nn.top_k(y_pred, k=top_k_pred)
        y_true_topk_t, y_true_topk_idx = K.tf.nn.top_k(y_true_rnd, k=top_k_true)
        
        batch_idx = K.tf.stack([K.tf.range(K.tf.shape(y_true)[0])]*top_k, axis=1)
        y_need_idx = K.tf.concat([y_pred_topk_idx, y_true_topk_idx], axis=1)
        y_need_idx2 = K.tf.stack([batch_idx, y_need_idx], axis=2)
        
        y_pred_topk = K.tf.gather_nd(params=y_pred, indices=y_need_idx2)
        y_true_topk = K.tf.gather_nd(params=y_true, indices=y_need_idx2)

        s1 = K.tf.stack([y_pred_topk]*top_k, axis=1)
        s2 = K.tf.stack([y_pred_topk]*top_k, axis=2)
        g1 = K.tf.stack([y_true_topk]*top_k, axis=1)
        g2 = K.tf.stack([y_true_topk]*top_k, axis=2)
 
        s1 -= s2
        g1 -= g2

        hinge = (1.0 - s1 * g1) * K.tf.abs(g1)
        hinge = K.tf.maximum(0.0, hinge)

        loss = K.sum(K.sum(hinge, axis=1), axis=1)

        return loss

    return loss_func

def logloss_with_2side_top(class_num=1999, top_k_pred = 200, top_k_true = 20):
    
    def loss_func(y_true, y_pred):
        top_k = top_k_pred + top_k_true
        
        y_true = K.tf.cast(y_true, tf.float32)
        y_true_rnd = y_true + tf.random_uniform(tf.shape(y_true), 0, 0.2)
        
        y_pred_topk_t, y_pred_topk_idx = K.tf.nn.top_k(y_pred, k=top_k_pred)
        y_true_topk_t, y_true_topk_idx = K.tf.nn.top_k(y_true_rnd, k=top_k_true)
        
        batch_idx = K.tf.stack([K.tf.range(K.tf.shape(y_true)[0])]*top_k, axis=1)
        y_need_idx = K.tf.concat([y_pred_topk_idx, y_true_topk_idx], axis=1)
        y_need_idx2 = K.tf.stack([batch_idx, y_need_idx], axis=2)
        
        y_pred_topk = K.tf.gather_nd(params=y_pred, indices=y_need_idx2)
        y_true_topk = K.tf.gather_nd(params=y_true, indices=y_need_idx2)

        loss = K.sum(K.binary_crossentropy(y_pred_topk, y_true_topk), axis=-1)

        return loss

    return loss_func

def logloss_with_2side_top_pn(class_num=1999, top_k_pos = 20, top_k_neg = 200):
    
    def loss_func(y_true, y_pred):
        top_k = top_k_pos + top_k_neg
        
        y_true = K.tf.cast(y_true, tf.float32)
        y_pos = y_true + y_pred * 0.1
        y_neg = 1.0 - y_true + y_pred * 0.1
        
        y_pos_topk_t, y_pos_topk_idx = K.tf.nn.top_k(y_pos, k=top_k_pos)
        y_neg_topk_t, y_neg_topk_idx = K.tf.nn.top_k(y_neg, k=top_k_neg)
        
        batch_idx = K.tf.stack([K.tf.range(K.tf.shape(y_true)[0])]*top_k, axis=1)
        y_need_idx = K.tf.concat([y_pos_topk_idx, y_neg_topk_idx], axis=1)
        y_need_idx2 = K.tf.stack([batch_idx, y_need_idx], axis=2)
        
        y_pred_topk = K.tf.gather_nd(params=y_pred, indices=y_need_idx2)
        y_true_topk = K.tf.gather_nd(params=y_true, indices=y_need_idx2)

        loss = K.sum(K.binary_crossentropy(y_pred_topk, y_true_topk), axis=-1)

        return loss

    return loss_func
