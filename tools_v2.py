# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:24:15 2024

@author: Admin
"""

# This is the class for the image rotation and clip

import numpy as np
import math
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from skimage.feature import peak_local_max
import ipywidgets as widgets
from ipywidgets import interact
import copy
import Gaussian_functions_v2 as gf
from scipy.ndimage import rotate

def image_preprocess(TEM_array, rotate_angle, normalized_maximum_intensity, low_intensity_filter, x, y, width, height, line_width):

    image = TEM_array
    rotated_im =  rotate(image, rotate_angle, order = 1)
    rect = rect = Rectangle((x,y), width, height, linewidth = line_width, edgecolor = 'red', facecolor = 'none')
    cliped_image = rotated_im[y:y+height, x:x+width]
    
    im_analyzed = cliped_image
    im_analyzed = im_analyzed - np.min(im_analyzed)
    im_analyzed = np.where(im_analyzed < low_intensity_filter*np.max(im_analyzed), 0, im_analyzed - low_intensity_filter*np.max(im_analyzed))
    
    im_analyzed = normalized_maximum_intensity*im_analyzed/np.max(im_analyzed)
    tf_im_analyzed = tf.convert_to_tensor(im_analyzed, dtype=tf.float32)       # For the futrue use with tensorflow
    
    plt.figure()
    plt.title("Rotated TEM image", pad = 20)
    plt.imshow(rotated_im , cmap = "grey")
    plt.gca().add_patch(rect)

    plt.show()
    
    plt.figure()
    plt.title("Processed TEM image")
    plt.imshow(im_analyzed, cmap = "grey")
    plt.show()

    return im_analyzed, tf_im_analyzed

class lattice_position():
    def __init__(self, x, y, x_off, y_off, len_pix, row, col, sliding):
        
        self.x = x
        self.y = y
        self.x_off = x_off
        self.y_off = y_off
        self.len_pix = len_pix
        self.row = row
        self.col = col
        self.sliding = sliding
        
    def lattices(self):

        x = np.arange(self.col)*self.x/self.len_pix + self.x_off
        y = np.arange(self.row)*self.y/self.len_pix + self.y_off

        X, Y = np.meshgrid(x, y)

        lattices = np.stack([X, Y], axis = 2)

        for i in range(self.row - 1):

            lattices[i+1, :, 0] += self.sliding[i]
        
        return np.round(lattices).astype(int)

    def boxes(self, lattices):

        patches = []

        for i in range(self.row):
            for j in range(self.col):

                x = lattices[i,j][0]
                y = lattices[i,j][1]
                
                patches.append(Rectangle((x, y), self.x/self.len_pix, self.y/self.len_pix))
                
        collection = PatchCollection(patches, edgecolor='black', facecolor = 'none')
        
        return collection

    def overlapping(self, image, lattices):

        dy = np.round(self.y/self.len_pix).astype(int)
        dx = np.round(self.x/self.len_pix).astype(int)

        image_array = np.zeros((self.row, self.col, dy, dx))

        for i in range(self.row):
            for j in range(self.col):
                
                image_array[i, j] = image[lattices[i, j][1] : lattices[i, j][1] + dy, 
                lattices[i, j][0] : lattices[i, j][0] + dx]

        sum_image = np.sum(image_array, axis = (0,1))      
                        
        return image_array, sum_image/(lattices.shape[0]*lattices.shape[1])

def lattice_parameter_tunning(x, y, x_off, y_off, len_pix, row, col, sliding, im_analyzed,
                             x_look, y_look, width_look, height_look):

    rect = Rectangle((x_look, y_look), width_look, height_look, 
                 linewidth = 2, edgecolor = 'red', facecolor = 'none')

    lat = lattice_position(x, y, x_off, y_off, len_pix, row, col, sliding)
    lattices = lat.lattices()
    image_array, unit_cell_image = lat.overlapping(im_analyzed, lattices)

    lattice_num = lattices.shape[0]*lattices.shape[1]
    
    fig, ax = plt.subplots(1, 3, figsize = (10, 3))
    
    ax[0].set_title("0 row x-scan for x check")
    ax[1].set_title("0 col y-scan for y check")
    ax[2].set_title("0 col x-scan for slidings check")
    
    for i in range(image_array.shape[1]):
    
        image = np.mean(image_array[0, i][y_look:y_look + height_look, x_look:x_look + width_look], 
                        axis = 0)
        n_image = image/np.max(image)
        
        ax[0].plot(n_image)
        print(f"{i} col peak positions for x tunning:", find_peaks(n_image)[0])
    
    print("\n")
    
    for i in range(image_array.shape[0]):
    
        image = np.mean(image_array[i, 0][y_look:y_look + height_look, x_look:x_look + width_look], 
                        axis = 1)
        n_image = image/np.max(image)
    
        ax[1].plot(n_image)
        print(f"{i} row peak positions for y tunning:", find_peaks(n_image)[0])
    
    print("\n")
    
    for i in range(image_array.shape[0]):
    
        image = np.mean(image_array[i, 0][y_look:y_look + height_look, x_look:x_look + width_look], 
                        axis = 0)
        n_image = image/np.max(image)
    
        ax[2].plot(n_image)
        print(f"{i} row peak positions for slidings tunning:", find_peaks(n_image)[0])
    
    print("\n")
    
    fig, ax =  plt.subplots(1, 1, figsize = (5, 5))
    
    ax.imshow(im_analyzed, cmap = "grey")   
    collection = lat.boxes(lattices)    
    ax.add_collection(collection)
    ax.set_title('TEM image with grid')

    TEM_image_with_grid = fig

    fig, ax =  plt.subplots(1, 1, figsize = (5, 5))
    
    ax.imshow(unit_cell_image, cmap = "grey")
    ax.add_patch(rect)
    ax.set_title('Unit cell image')
    
    plt.show()

    tf_unit_cell_image = tf.cast(unit_cell_image, dtype = tf.float32)
    
    return unit_cell_image, tf_unit_cell_image, lattices, lattice_num, TEM_image_with_grid


def unit_cell_investigator(unit_cell_image):

    def plot_interact(horizon, vertical):
    
        horizon = int(horizon)
        vertical = int(vertical)
        fig, ax = plt.subplots(1, 3, figsize = (10, 5))
        fig.suptitle("TEM image of average unit cell and intensity investigator")
        ax[0].imshow(unit_cell_image, cmap = "grey")
        ax[0].axhline(y = horizon, color = 'red', linewidth = 1)
        ax[0].axvline(x = vertical, color = 'blue', linewidth = 1)
    
    
        ax[1].plot(unit_cell_image[horizon,:])
        ax[1].set_title("horizontal")
        ax[1].set_xlim(0, unit_cell_image.shape[1]-1)
        ax[2].plot(unit_cell_image[:,vertical])
        ax[2].set_title("vertical")
        ax[2].set_xlim(0, unit_cell_image.shape[0]-1)
    
    interact(plot_interact, horizon=widgets.FloatSlider(min=0, max=unit_cell_image.shape[0]-1, step=1, value=0), 
             vertical=widgets.FloatSlider(min=0, max=unit_cell_image.shape[1]-1, step=1, value=0));
    
    plt.show()

def atom_positions_iu_gen_check(Atom_positions_dic, unit_cell_image, x, y, len_pix, marker_size):

    atom_type_num = len(Atom_positions_dic)
    posit_pix = copy.deepcopy(list(Atom_positions_dic.values()))
    atom_num_list = [len(posit_pix[i]) for i in range(atom_type_num)]

    atom_num_in_unit_cell = 0

    for i in range(len(atom_num_list)):

        atom_num_in_unit_cell += atom_num_list[i]
    
    plt.figure()
    plt.title("Atomic position looks vaild?", pad = 15)
    plt.imshow(unit_cell_image, cmap = "gray")
    
    for atom in range(len(posit_pix)):
    
        posit_pix[atom] = np.array(posit_pix[atom])
    
        posit_pix[atom][:,0] = posit_pix[atom][:,0]*int(x/len_pix)
        posit_pix[atom][:,1] = posit_pix[atom][:,1]*int(y/len_pix)
    
        posit_pix[atom] = posit_pix[atom].astype(int)
    
        plt.scatter(posit_pix[atom][:,0], posit_pix[atom][:,1], s = marker_size)
        
    return posit_pix, atom_num_list, atom_num_in_unit_cell

def unit_cell_peak_finder(peak_finder_pad_size_list, 
                          atom_num_list, posit_pix, unit_cell_image, marker_size, atom_name_list):
    
    posit_pix_peaks = []
    
    atom_num_list = np.array(atom_num_list).astype(np.int64)

    
    
    for i in range(len(atom_num_list)):
    
        peaks = []
    
        peak_ra = np.zeros((atom_num_list[i], 2))
        
        for j in range(atom_num_list[i]):

            none_count = 0
    
            cen_x = np.round(posit_pix[i][j,0]).astype(np.int64)
            cen_y = np.round(posit_pix[i][j,1]).astype(np.int64)
            r = peak_finder_pad_size_list[i]
    
            peak = peak_local_max(unit_cell_image[cen_y - r: cen_y + r + 1,
                                        cen_x - r: cen_x + r + 1],
                                        num_peaks = 1)

            if len(peak) == 0:
    
                peak_ra[j][0] = posit_pix[i][j,0]
                peak_ra[j][1] = posit_pix[i][j,1]
    
                none_count +=1

            else:
            
                peak_ra[j][0] = peak[0][1]- r + cen_x
                peak_ra[j][1] = peak[0][0]- r + cen_y

            print(f"{atom_name_list[i]}-{j} non counts: {none_count}")
        
        peaks = np.array(peak_ra)
    
        posit_pix_peaks.append(peaks)
    
    plt.imshow(unit_cell_image, cmap = "grey")

    print("")
    print("posit_pix_inegers:\n")
    print(posit_pix)
    print("\n")
    print("posit_pix_peaks:\n")
    print(posit_pix_peaks)
    
    for i in range(len(atom_num_list)):
    
        plt.scatter(posit_pix_peaks[i].T[0], posit_pix_peaks[i].T[1], s = marker_size)

    return posit_pix_peaks
        
def unpack_atom_type(packed_array, lattice_num, atom_num_list):

    atom_type_num = len(atom_num_list)
    unpacked_array = []
    
    accul = 0

    for i in range(atom_type_num):

        unpacked_array.append(packed_array[:, accul : accul + lattice_num*atom_num_list[i]])
        accul += lattice_num*atom_num_list[i]

    return unpacked_array


def unit_cell_optimization_result(posit_pix, atom_num_list, pad_size_list, unit_cell_image, unit_cell_image_shape, 
                                 atom_positions_in_unit_cell_file_name, unit_cell_params_file_name, sample, atom_name_list,
                                 unit_cell_information_file_name, marker_size, base_dir):

    dposit = np.load(base_dir + atom_positions_in_unit_cell_file_name + "_" + sample + ".npy")
    unit_cell_params = np.load(base_dir + unit_cell_params_file_name + "_" + sample + ".npy")
    
    tfv_dposit = tf.Variable(tf.cast(dposit, dtype = tf.float32))
    tfv_unit_cell_params = tf.Variable(tf.cast(unit_cell_params, dtype = tf.float32))
    
    positions = np.concatenate(posit_pix, axis = 0)
    positions = np.transpose(positions)
    
    atom_resolved_positions = unpack_atom_type(positions+dposit, 1, atom_num_list)
    atom_resolved_params = unpack_atom_type(unit_cell_params, 1, [1]*len(atom_num_list))

    Unit_cell_markers = plt.figure()
        
    plt.imshow(unit_cell_image, cmap = "gray")
    plt.title("Unit cell w/ markers")

    unit_cell_information = ""
    
    for atom in range(len(atom_num_list)):
        
        plt.scatter(atom_resolved_positions[atom][0,:], 
                    atom_resolved_positions[atom][1,:], 
                    s = marker_size,
                    label = atom_name_list[atom])

        atomic_positions_spring = "Atomic positions of " + atom_name_list[atom] + ":\n" + f"{atom_resolved_positions[atom].T}" + "\n"
        params_spring = "Gaussian params of " + atom_name_list[atom] + ":\n" + f"{atom_resolved_params[atom].T}" + "\n"
        
        print(atomic_positions_spring)
        print(params_spring)

        unit_cell_information += atomic_positions_spring
        unit_cell_information += params_spring

    plt.legend(loc = "lower left",
              bbox_to_anchor = (1.02, 0),
              frameon=False)
    
    Unit_cell_comparison, ax = plt.subplots(1, 2, figsize = (10, 5))

    simul_image = gf.Gaussian_unitcell(posit_pix, tfv_dposit, tfv_unit_cell_params, atom_num_list, pad_size_list, 
                                       unit_cell_image_shape)
    
    ax[0].imshow(unit_cell_image, cmap = "gray")
    ax[0].set_title("Unit cell")
    ax[1].imshow(simul_image, cmap = "gray")
    ax[1].set_title("Simulated unit cell")

    with open(base_dir + unit_cell_information_file_name + "_" + sample + ".txt", "w") as file:
        file.write(unit_cell_information)

    return (tfv_dposit, tfv_unit_cell_params, atom_resolved_positions, atom_resolved_params, simul_image,
            Unit_cell_markers, Unit_cell_comparison)


def positions_params_gen(lattices, posit_pix, tfv_dposit, tfv_unit_cell_params, lattice_num, atom_num_list):

    total_num = (np.sum(np.array(atom_num_list))*lattice_num + 1e-8).astype(np.int64)
    lattices = lattices.reshape(-1, 2)
        
    positions = np.zeros((2, total_num))
    params = np.zeros((tfv_unit_cell_params.shape[0], total_num))
    
    accul = 0
    accul_atom = 0
          
    for atom in range(len(atom_num_list)):
        for lattice in range(lattice_num):

            positions[:, lattice*atom_num_list[atom] + accul : (lattice+1)*atom_num_list[atom] + accul] = \
            lattices[lattice][:,None] + posit_pix[atom].T  + np.array(tfv_dposit)[:, accul_atom: accul_atom + atom_num_list[atom]]
            
            params[:, lattice*atom_num_list[atom] + accul : (lattice+1)*atom_num_list[atom] + accul] =\
            np.array(tfv_unit_cell_params)[:, atom][:,None]
        
        accul += lattice_num*atom_num_list[atom]
        accul_atom += atom_num_list[atom]                    

    return positions, params

def positions_correction(positions, posit_pix_peaks, 
                                               lattices, anchor_atom, anchor_atom_pad,
                                                     atom_num_list, lattice_num, 
                                               im_analyzed, marker_size, row, col,
                        tfv_dposit):

    atom_num_list = np.array(atom_num_list).astype(np.int64)
    lattice_num = np.array(lattice_num).astype(np.int64)
    
    atom = anchor_atom[0]
    order = anchor_atom[1]
    
    start = lattice_num*np.sum(atom_num_list[:atom])
    end = start + lattice_num*atom_num_list[atom]
    
    anchor_total = positions[:, start:end]
    anchor_total = anchor_total.reshape(2, lattice_num, atom_num_list[atom])
    
    anchor = anchor_total[:, :, order]
    
    anchor_peaks = np.zeros(anchor.shape)
    
    for i in range(anchor.shape[1]):
    
        
        cen_x =  np.round(anchor[0, i]).astype(np.int64)
        cen_y =  np.round(anchor[1, i]).astype(np.int64)
        r = anchor_atom_pad
    
        peak = peak_local_max(im_analyzed[cen_y - r: cen_y + r + 1,
                                         cen_x - r: cen_x + r + 1],
                                         num_peaks = 1)
    
        anchor_peaks[0, i] = peak[0][1] - r + cen_x
        anchor_peaks[1, i] = peak[0][0] - r + cen_y
    
    plt.figure()
    plt.imshow(im_analyzed, cmap = 'grey')
    plt.title("Anchors look vaild?")
    plt.scatter(anchor_peaks[0,:], anchor_peaks[1,:], s = marker_size)
    
    anchor_lattices = anchor.T.reshape(row, col, 2)
    anchor_peaks_lattices = anchor_peaks.T.reshape(row, col, 2)
    
    del_lattices = anchor_peaks_lattices - anchor_lattices
    
    lattices_c = lattices + del_lattices
    
    total_num = (np.sum(np.array(atom_num_list))*lattice_num + 1e-8).astype(np.int64)
    lattices_c = lattices_c.reshape(-1, 2)
        
    positions_c = np.zeros((2, total_num))
    
    accul = 0
    accul_atom = 0
          
    for atom in range(len(atom_num_list)):
        for lattice in range(lattice_num):
    
            positions_c[:, lattice*atom_num_list[atom] + accul : (lattice+1)*atom_num_list[atom] + accul]\
            = lattices_c[lattice][:,None] + posit_pix_peaks[atom].T
        
        accul += lattice_num*atom_num_list[atom]
        accul_atom += atom_num_list[atom]   

    return positions_c, lattices_c


def positions_peak_finder(positions_c, atom_num_list, lattice_num, peak_finder_pad_size_list, 
                          im_analyzed, atom_name_list):

    positions_peak = np.zeros(positions_c.shape)
    
    atom_num_list = np.array(atom_num_list).astype(np.int64)
    lattice_num = np.array(lattice_num).astype(np.int64)
    
    accul = 0
    
    
    for atom in range(len(atom_num_list)):

        none_count = 0
    
        for i in range(atom_num_list[atom]*lattice_num):
    
            cen_x = np.round(positions_c[0, accul]).astype(np.int64)
            cen_y = np.round(positions_c[1, accul]).astype(np.int64)
            r = peak_finder_pad_size_list[atom]
                
            peak = peak_local_max(im_analyzed[cen_y - r: cen_y + r,
                                        cen_x - r: cen_x + r],
                                        num_peaks = 1)
    
            if len(peak) == 0:
    
                positions_peak[0, accul] = positions_c[0, accul]
                positions_peak[1, accul] = positions_c[1, accul]
    
                none_count +=1
    
            else:
                
                positions_peak[0, accul] = peak[0][1] - r + cen_x
                positions_peak[1, accul] = peak[0][0] - r + cen_y
    
            accul += 1

        print(f"{atom_name_list[atom]} non counts: {none_count}")

    return positions_peak


def free_atom_mask(lattice_num, atom_num_list, Fixed_params):
    
    lattice_num = tf.cast(lattice_num, dtype = tf.int32)
    mask_params = tf.zeros((0, 2), dtype = tf.int32)
    
    for i in range(len(Fixed_params)):
    
        atom_index = Fixed_params[i][1]
    
        before_atom = 0
    
        for j in range(atom_index):
    
            before_atom += atom_num_list[j]*lattice_num
        
        atom_index_array = tf.range(before_atom, before_atom + atom_num_list[atom_index]*lattice_num)
        param_index_array = tf.fill([len(atom_index_array),], Fixed_params[i][0])
    
        mask_params_part = tf.stack([param_index_array, atom_index_array], axis = 1)
    
        mask_params = tf.concat([mask_params, mask_params_part], axis = 0)
    
    return mask_params


def free_atom_opmization_results(base_dir, dpositions_file_name, params_file_name,
                                      positions, lattice_num, atom_num_list, pad_size_list, 
                                      im_analyzed, marker_size, sample):

    
    im_shape = im_analyzed.shape
    
    params = np.load(base_dir + params_file_name + "_" + sample +".npy")
    dpositions = np.load(base_dir + dpositions_file_name + "_" + sample + ".npy")
    
    tfv_params = tf.Variable(params, dtype = tf.float32)
    tfv_dpositions = tf.Variable(dpositions, dtype = tf.float32)
    
    atom_resolved_positions = unpack_atom_type(positions+dpositions, lattice_num, atom_num_list)
    atom_resolved_params = unpack_atom_type(params, lattice_num, atom_num_list)
    
    plt.imshow(im_analyzed, cmap = "gray")
    plt.title("TEM image with markers")
    
    for atom in range(len(atom_num_list)):
        
        plt.scatter(atom_resolved_positions[atom][0,:], atom_resolved_positions[atom][1,:], s = marker_size)
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 5))
    
    ax[0].imshow(im_analyzed, cmap = "gray")
    ax[0].set_title("TEM image")
    ax[1].imshow(gf.Gaussian_np_fine(positions, dpositions, params, 
                                           atom_num_list, pad_size_list, im_shape), cmap = "gray")
    ax[1].set_title("Simulated TEM image")

    return tfv_params, tfv_dpositions, atom_resolved_positions, atom_resolved_params

def boundary_cut(atom_resolved_positions, atom_resolved_params, row, col, cut_row = 0, cut_col = 0):

    lattice_num = row*col

    cut_atom_positions = []
    cut_atom_params = []

    for atom in range(len(atom_resolved_positions)):
        
        total_num = int(atom_resolved_positions[atom].shape[1])
        
        atoms_in_lattice = int(total_num/lattice_num)
    
        cut_row_indices_top = np.arange(0 , cut_row*atoms_in_lattice*col) 
        cut_row_indices_botom = np.arange(total_num - cut_row*atoms_in_lattice*col , total_num)
        cut_row_indices = np.concatenate([cut_row_indices_top, cut_row_indices_botom])

        posit_row_cut = np.delete(atom_resolved_positions[atom], cut_row_indices, axis = 1)
        par_row_cut = np.delete(atom_resolved_params[atom], cut_row_indices, axis = 1)

        cut_col_indices_list = []

        atoms_in_row = col*atoms_in_lattice

        for i in range(row):
    
            cut_col_indices_list.append(np.arange(i*atoms_in_row, i*atoms_in_row + cut_col*atoms_in_lattice))
            cut_col_indices_list.append(np.arange((i+1)*atoms_in_row - cut_col*atoms_in_lattice, (i+1)*atoms_in_row))
    
        cut_col_indices = np.concatenate(cut_col_indices_list)
    
        cut_atom_positions.append(np.delete(posit_row_cut, cut_col_indices, axis = 1)) 
        cut_atom_params.append(np.delete(par_row_cut, cut_col_indices, axis = 1))
    
    return cut_atom_positions, cut_atom_params

