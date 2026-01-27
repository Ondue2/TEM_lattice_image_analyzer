# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:57:27 2024

@author: Admin
"""

import tensorflow as tf
import numpy as np

def compute_loss_tf(given_image, simulated_image):

    fit_loss = tf.reduce_sum(tf.where(simulated_image != 0, tf.square(given_image - simulated_image), 0))
    count_num = tf.math.count_nonzero(simulated_image, dtype=tf.float32)
   
    return fit_loss/count_num

def compute_loss_np(given_image, simulated_image):

    fit_loss = np.sum(np.where(simulated_image != 0, np.square(given_image - simulated_image), 0))
    count_num = np.count_nonzero(simulated_image)

    loss = fit_loss/count_num 

    return loss

