# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:55:00 2024

@author: Admin
"""

import numpy as np
import tensorflow as tf
from math import pi


def Gaussian_fine(positions, tfv_dposition, tfv_params, atom_num_list, pad_size_list, im_shape):

    pad_array_list = []
    slices_list = []
    slices = 0

    lattice_num = positions.shape[1]/np.sum(np.array(atom_num_list))

    atom_type_num = len(atom_num_list)

    add_pix = np.max(pad_size_list)
    
    pad_size_array = np.repeat(np.array(pad_size_list), repeats = (np.array(atom_num_list)*lattice_num).astype(np.int64), axis = 0)

    tf_white_board = tf.zeros((im_shape[0] + 2*add_pix, im_shape[1] + 2*add_pix), dtype = tf.float32)

    for i in range(positions.shape[1]):

        pad_size = pad_size_array[i]
        x_pad = tf.range(2*pad_size+1)
        y_pad = tf.range(2*pad_size+1)
        
        X, Y = tf.meshgrid(x_pad, y_pad)

        X = tf.reshape(X, [-1])
        Y = tf.reshape(Y, [-1])
        
        tf_pad = tf.cast(tf.stack([X, Y], axis = 1), dtype = tf.float32)

        tf_x = tf.cast(positions[0,i], tf.int32) + add_pix
        tf_y = tf.cast(positions[1,i], tf.int32) + add_pix
        tf_A = tf.cast(tfv_params[0,i], tf.float32)
        tf_sig_x = tf.cast(tfv_params[1,i], tf.float32)
        tf_sig_y = tf.cast(tfv_params[2,i], tf.float32)
        tf_theta = tf.cast(tfv_params[3,i], tf.float32)*pi/180

        tf_dx = tf_pad[:,0] - pad_size - tfv_dposition[0,i]
        tf_dy = tf_pad[:,1] - pad_size - tfv_dposition[1,i]

        tf_dxp = tf_dx*tf.cos(tf_theta) + tf_dy*tf.sin(tf_theta)
        tf_dyp = -tf_dx*tf.sin(tf_theta) + tf_dy*tf.cos(tf_theta)
   
        tf_array_value = tf_A*tf.exp(-(tf_dxp**2/(2*tf_sig_x**2)) -(tf_dyp**2/(2*tf_sig_y**2)))

        tf_X, tf_Y = tf.meshgrid(tf.range(tf_x - pad_size, tf_x + pad_size + 1), tf.range(tf_y - pad_size, tf_y + pad_size + 1))
        tf_flatX = tf.reshape(tf_X, -1)
        tf_flatY = tf.reshape(tf_Y, -1)
        indices = tf.stack([tf_flatY, tf_flatX], axis = -1) 
         
        tf_white_board = tf.tensor_scatter_nd_add(tf_white_board, indices, tf_array_value)
        
    cliped_tf_white_board = tf_white_board[add_pix : tf_white_board.shape[0] - add_pix, add_pix : tf_white_board.shape[1] - add_pix]

    return  cliped_tf_white_board

def Gaussian_np_fine(positions, dpositions, tfv_params, atom_num_list, pad_size_list, im_shape):

    np_params = np.array(tfv_params)

    pad_array_list = []
    slices_list = []
    slices = 0

    lattice_num = positions.shape[1]/np.sum(np.array(atom_num_list))

    atom_type_num = len(atom_num_list)

    add_pix = np.max(pad_size_list)
    
    pad_size_array = np.repeat(np.array(pad_size_list), repeats = (np.array(atom_num_list)*lattice_num).astype(np.int64), axis = 0)

    white_board = np.zeros((im_shape[0] + 2*add_pix, im_shape[1] + 2*add_pix))

    for i in range(positions.shape[1]):

        pad_size = pad_size_array[i]
        
        x_pad = np.arange(2*pad_size+1)
        y_pad = np.arange(2*pad_size+1)

        X, Y = np.meshgrid(x_pad, y_pad)

        pad = np.column_stack([X.ravel().astype(int), Y.ravel().astype(int)])
        
        x_cor = int(positions[0,i]) + add_pix
        y_cor = int(positions[1,i]) + add_pix
        A = np_params[0,i]
        sig_x = np_params[1,i]
        sig_y = np_params[2,i]
        theta = np_params[3,i]*pi/180

        dx = pad[:,0] - pad_size - dpositions[0,i]
        dy = pad[:,1] - pad_size - dpositions[1,i]

        dxp = dx*np.cos(theta) + dy*np.sin(theta)
        dyp = -dx*np.sin(theta) + dy*np.cos(theta)
   
        array_value = A*np.exp(-(dxp**2/(2*sig_x**2)) -(dyp**2/(2*sig_y**2)))

        pad_value = array_value.reshape(2*pad_size+1, 2*pad_size+1)

        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] = \
        white_board[y_cor - pad_size:y_cor + pad_size + 1, x_cor - pad_size:x_cor + pad_size + 1] + pad_value

    cliped_white_board = white_board[add_pix : white_board.shape[0] - add_pix, add_pix : white_board.shape[1] - add_pix]

    return  cliped_white_board

def Gaussian_unitcell(posit_pix, tfv_dposit, tfv_unit_cell_params, atom_num_list, pad_size_list, figsize):

    positions = np.concatenate(posit_pix, axis = 0)
    positions = np.transpose(positions)

    params = tf.repeat(tfv_unit_cell_params, repeats = atom_num_list, axis = 1)

    add_pix = np.max(pad_size_list)

    pad_size_array = np.repeat(np.array(pad_size_list), repeats = atom_num_list, axis = 0)

    tf_white_board = tf.zeros((figsize[0] + 2*add_pix, figsize[1] + 2*add_pix), dtype = tf.float32)

    for i in range(positions.shape[1]):

        pad_size = pad_size_array[i]
        x_pad = tf.range(2*pad_size+1)
        y_pad = tf.range(2*pad_size+1)
        
        X, Y = tf.meshgrid(x_pad, y_pad)

        X = tf.reshape(X, [-1])
        Y = tf.reshape(Y, [-1])
        tf_pad = tf.cast(tf.stack([X, Y], axis = 1), dtype = tf.float32)

        tf_x = tf.cast(positions[0,i], tf.int32) + add_pix
        tf_y = tf.cast(positions[1,i], tf.int32) + add_pix
        tf_A = tf.cast(params[0,i], tf.float32)
        tf_sig_x = tf.cast(params[1,i], tf.float32)
        tf_sig_y = tf.cast(params[2,i], tf.float32)
        tf_theta = tf.cast(params[3,i], tf.float32)*pi/180

        tf_dx = tf_pad[:,0] - pad_size - tfv_dposit[0,i]
        tf_dy = tf_pad[:,1] - pad_size - tfv_dposit[1,i]

        tf_dxp = tf_dx*tf.cos(tf_theta) + tf_dy*tf.sin(tf_theta)
        tf_dyp = -tf_dx*tf.sin(tf_theta) + tf_dy*tf.cos(tf_theta)
   
        tf_array_value = tf_A*tf.exp(-(tf_dxp**2/(2*tf_sig_x**2)) -(tf_dyp**2/(2*tf_sig_y**2)))

        tf_X, tf_Y = tf.meshgrid(tf.range(tf_x - pad_size, tf_x + pad_size + 1), tf.range(tf_y - pad_size, tf_y + pad_size + 1))
        tf_flatX = tf.reshape(tf_X, -1)
        tf_flatY = tf.reshape(tf_Y, -1)
        indices = tf.stack([tf_flatY, tf_flatX], axis = -1) 
         
        tf_white_board = tf.tensor_scatter_nd_add(tf_white_board, indices, tf_array_value)

    cliped_tf_white_board = tf_white_board[add_pix : tf_white_board.shape[0] - add_pix, add_pix : tf_white_board.shape[1] - add_pix]
    
    cwb_h, cwb_w = cliped_tf_white_board.shape

    add_pix = tf.cast(add_pix, dtype = tf.int32)

    up_x, up_y = tf.meshgrid(tf.range(0, cwb_w), tf.range(0, add_pix))
    down_x, down_y = tf.meshgrid(tf.range(0, cwb_w), tf.range(cwb_h - add_pix, cwb_h))
    left_x, left_y = tf.meshgrid(tf.range(0, add_pix), tf.range(0, cwb_h))
    right_x, right_y = tf.meshgrid(tf.range(cwb_w - add_pix, cwb_w), tf.range(0, cwb_h))
    upleft_x, upleft_y = tf.meshgrid(tf.range(0, add_pix), tf.range(0, add_pix))
    upright_x, upright_y = tf.meshgrid(tf.range(cwb_w - add_pix, cwb_w), tf.range(0, add_pix))
    downleft_x, downleft_y = tf.meshgrid(tf.range(0, add_pix), tf.range(cwb_h - add_pix, cwb_h))
    downright_x, downright_y = tf.meshgrid(tf.range(cwb_w - add_pix, cwb_w), tf.range(cwb_h - add_pix, cwb_h))

    up_x_f, up_y_f = tf.reshape(up_x, -1), tf.reshape(up_y, -1)
    down_x_f, down_y_f = tf.reshape(down_x, -1), tf.reshape(down_y, -1)
    left_x_f, left_y_f = tf.reshape(left_x, -1), tf.reshape(left_y, -1)
    right_x_f, right_y_f = tf.reshape(right_x, -1), tf.reshape(right_y, -1)
    upleft_x_f, upleft_y_f = tf.reshape(upleft_x, -1), tf.reshape(upleft_y, -1)
    upright_x_f, upright_y_f = tf.reshape(upright_x, -1), tf.reshape(upright_y, -1)
    downleft_x_f, downleft_y_f = tf.reshape(downleft_x, -1), tf.reshape(downleft_y, -1)
    downright_x_f, downright_y_f = tf.reshape(downright_x, -1), tf.reshape(downright_y, -1)

    up_indices = tf.stack([up_y_f, up_x_f], axis = -1)
    down_indices = tf.stack([down_y_f, down_x_f], axis = -1)
    left_indices = tf.stack([left_y_f, left_x_f], axis = -1)
    right_indices = tf.stack([right_y_f, right_x_f], axis = -1)
    upleft_indices = tf.stack([upleft_y_f, upleft_x_f], axis = -1)
    upright_indices = tf.stack([upright_y_f, upright_x_f], axis = -1)
    downleft_indices = tf.stack([downleft_y_f, downleft_x_f], axis = -1)
    downright_indices = tf.stack([downright_y_f, downright_x_f], axis = -1)

    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, up_indices, tf.reshape(tf_white_board[-add_pix:, add_pix:-add_pix], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, down_indices, tf.reshape(tf_white_board[:add_pix, add_pix:-add_pix], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, left_indices, tf.reshape(tf_white_board[add_pix:-add_pix, -add_pix:], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, right_indices, tf.reshape(tf_white_board[add_pix:-add_pix, :add_pix], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, upleft_indices, tf.reshape(tf_white_board[-add_pix:, -add_pix:], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, upright_indices, tf.reshape(tf_white_board[-add_pix:, :add_pix], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, downleft_indices, tf.reshape(tf_white_board[:add_pix, -add_pix:], [-1]))
    cliped_tf_white_board = tf.tensor_scatter_nd_add(cliped_tf_white_board, downright_indices, tf.reshape(tf_white_board[:add_pix, :add_pix], [-1]))

    return  cliped_tf_white_board

