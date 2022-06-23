#!/usr/bin/env python3

import os
import glob
import shutil
import datetime
import operator
import sys
import copy
import warnings

import scipy
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage import feature
from matplotlib import ticker
import cv2
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def segment_muscle_and_fat(water_image_data, fat_image_data,
                           min_shape_area=100, min_edge_length=20, n_erode_muscle=0,
                           n_fat_preserve = 4, n_erode_fat = 0, n_edge_dilate=5,
                           prop_thresh=0.5, solidity_thresh=0.1):
    """
    

    Parameters
    ----------
    water_image_data : TYPE
        np array of the the raw ppm-averaged water-weighted z-spectrum.
    fat_image_data : TYPE
         np array of the the raw ppm-averaged fat-weighted z-spectrum.
    edge_ratio_low : TYPE, optional
        DESCRIPTION. The default is 0.02.
    edge_ratio_high : TYPE, optional
        DESCRIPTION. The default is 0.05.
    min_shape_area : TYPE, optional
        DESCRIPTION. The default is 100.
    min_edge_length : TYPE, optional
        DESCRIPTION. The default is 20.
    n_erode_muscle : TYPE, optional
        DESCRIPTION. The default is 1.
    n_fat_preserve : TYPE, optional
        DESCRIPTION. The default is 2.
    n_erode_fat : TYPE, optional
        DESCRIPTION. The default is 1.
    n_edge_dilate : TYPE, optional
        DESCRIPTION. The default is 7.
    prop_thresh : TYPE, optional
        DESCRIPTION. The default is 0.2.
    solidity_thresh : TYPE, optional
        DESCRIPTION. The default is 0.1.

    Returns
    -------
    A list of the water (muscle) mask, fat mask, combined mask and intermediate products
    in that order. The intermediate products are stored as a dictionary:
        'fat_mask_raw': simple otsu-threshholded fat mask
        'water_mask_raw': simple otsu-threshholded water mask
        'combined_mask_raw': simple combination mask without removing extraneous pixels
        'thetas': a np array of fat-weighted slope directions
        'edge_mask': a np array of fat-weighted edges
        'hole_mask': a np array of arm bone
        'flat_intersect': z-flattened intersection of dilated edges
        'raw_skeleton': the raw skeleton of the flat_intersect
        'skeleton': the skeleton of the flat_intersect with dangling segments removed

    """
    intermediates = {}
    
    fat_thresh = threshold_otsu(fat_image_data)
    fat_mask = fat_image_data > fat_thresh
    
    intermediates['fat_mask_raw'] = fat_mask.copy()

    water_thresh = threshold_otsu(water_image_data)
    water_mask = water_image_data > water_thresh
    #water_mask = morphology.closing(water_mask)
    
    
    
        
    # get fat edges
    edge_mask = np.zeros_like(fat_image_data)
    for i in range(edge_mask.shape[2]):
        sli = fat_image_data[:,:,i]
        
        wm = water_mask[:,:,i]
        fm = fat_mask[:,:,i]
        
        edges = feature.canny(sli, sigma=2)
        edge_mask[:,:,i] = edges
        
    
    # overlay
    
    # first, fat beats water
    water_mask[fat_mask > 0] = 0
    
    intermediates['water_mask_raw'] = water_mask.copy()
    
    
    combined_mask = np.zeros_like(fat_mask)
    
    combined_mask = combined_mask + fat_mask
    combined_mask = combined_mask + water_mask*2
    # end result: 0 = nothing, 1 = fat, 2 = muscle, 3 = overlap
    intermediates['combined_mask_raw'] = combined_mask.copy()
            
    
    thetas = np.zeros_like(fat_image_data)
    for i in range(edge_mask.shape[2]):
        sli = fat_image_data[:,:,i]
        dy,dx = np.gradient(sli)
        theta = np.degrees(np.arctan2(dy,dx))
        thetas[:,:,i] = theta
        
    intermediates['thetas'] = thetas

   
    bulk_mask = combined_mask > 0
    center_x = bulk_mask.shape[0] / 2
    center_y = bulk_mask.shape[1] / 2
    zeroes_mask = np.zeros_like(bulk_mask).astype(bool)
    
    # find the arm muscle
    for i in range(bulk_mask.shape[2]):
        sli = water_mask[:,:,i]
        
        # erode n times, keep shapes with area > min_shape_area, select shape with centroid closest to center, dilate n times
        for j in range(n_erode_muscle):
            sli = morphology.binary_erosion(sli)
        labeled = measure.label(sli)
        
        try:
            props = measure.regionprops_table(labeled, properties={'label','area','centroid'})
            
            #max_area = max(props['area'])
            #max_index = list(props['area']).index(max_area)
            #max_label = props['label'][max_index]
            
            keep_labels, keep_areas, keep_centroids, dists_to_center = [], [], [], []
            
            for label, area, c1, c2 in zip(props['label'], props['area'], props['centroid-0'],props['centroid-1']):
                if area > min_shape_area:
                    keep_labels.append(label)
                    keep_areas.append(area)
                    keep_centroids.append((c1,c2))
                    
                    dist = abs( ( (c1-center_x)**2 + (c2-center_y)**2 )**0.5 )
                    
                    dists_to_center.append(dist)
                    
            min_dist = min(dists_to_center)
            min_index = dists_to_center.index(min_dist)
            min_label = keep_labels[min_index]
            
            sli[labeled != min_label] = 0
            for j in range(n_erode_muscle):
                sli = morphology.binary_dilation(sli)
            
            if n_erode_muscle > 0:
                sli = morphology.binary_closing(sli)
            
            water_mask[:,:,i] = sli
        except (IndexError, ValueError):
            water_mask[:,:,i] = zeroes_mask[:,:,i].copy() # happens when there are (no regions, no regions area > min_shape_area)
            
    
    # find bone
    found_hole = True
    hole_mask = zeroes_mask.copy()
    for i in range(bulk_mask.shape[2]):
        sli = water_mask[:,:,i].copy()
        sli = morphology.binary_dilation(sli) # dilate it once to close small channels
        
        filled = binary_fill_holes(sli)
        reverser = (filled.astype(int) - sli.astype(int)).astype(bool)
        
        labeled = measure.label(reverser)
        
        # same thing again, we'll pick the hole closest to the center given it's big enough
        # but here we'll just store the coordinates of the centroid
        try:
            props = measure.regionprops_table(labeled, properties={'label','area','centroid'})
            
            max_area = max(props['area'])
            max_index = list(props['area']).index(max_area)
            max_label = props['label'][max_index]
            
            keep_labels, keep_areas, keep_centroids, dists_to_center = [], [], [], []
            
            win_dist = np.inf
            win_cx, win_cy = None, None
            for label, area, c1, c2 in zip(props['label'], props['area'], props['centroid-0'],props['centroid-1']):
                if area > 10:
                    keep_labels.append(label)
                    keep_areas.append(area)
                    keep_centroids.append((c1,c2))
                    
                    dist = abs( ( (c1-center_x)**2 + (c2-center_y)**2 )**0.5 )
                    
                    dists_to_center.append(dist)
                    
            min_dist = min(dists_to_center)
            min_index = dists_to_center.index(min_dist)
            min_label = keep_labels[min_index]
            min_cent = keep_centroids[min_index]
            
            reverser[labeled != min_label] = 0
            
            reverser = morphology.binary_dilation(reverser)
            
            hole_mask[:,:,i] = reverser
            
            
            
        except (IndexError, ValueError):
            found_hole = False
            continue
    
    
    
    # remove fat edges from fat, but only where there is no arm muscle - will help with finding arm
    edge_mask = edge_mask.astype(int)
    edge_mask[morphology.binary_dilation(water_mask).astype(int) > 0] = 0
    
    #remove little short blibs of edge from the edge mask
    for i in range(edge_mask.shape[2]):
        sli = edge_mask[:,:,i]
        edge_lab = measure.label(sli)
        edge_props = measure.regionprops_table(edge_lab, properties={'label','area'})
        
        sorted_areas = edge_props['area'].copy()
        sorted_areas.sort()
        
        if len(sorted_areas) < 20:
            continue
        
        thresh_area = sorted_areas[-5]
        
        for label, area in zip(edge_props['label'], edge_props['area']):
            edit_label_locs = edge_lab == label
            
            if area < thresh_area: # get rid of it
                the_val = -1
            else:
                the_val = label
            
            sli[edit_label_locs == 1] = the_val
            #remove_label_locs = edge_lab == label
            #sli[remove_label_locs == 1] = area + 20
        edge_mask[:,:,i] = sli
    
    fat_mask[edge_mask >= 1] = 0
    
    
    # find the arm fat
    #n_fat_preserve = 2 # number of dilations around arm muscle to preserve fat
    #n_erode_fat = 1
    for i in range(bulk_mask.shape[2]):
        sli = fat_mask[:,:,i].astype(int)
        
        arm_slice = water_mask[:,:,i].copy().astype(int) # we'll use this to preserve fat around the arm muscle
        preserve_area = arm_slice.copy()
        for k in range(n_fat_preserve):
            preserve_area = morphology.binary_dilation(preserve_area)
        preserve_area = preserve_area.astype(int)
        preserve_area = preserve_area - arm_slice
        preserve_area[preserve_area == 0] = -1 # do this to help the equality statement in a few lines; would require more code if we didn't do this
        
        #sys.exit()
        # erode n times, keep shapes with area > min_shape_area, select shape with centroid closest to center, dilate n times
        for j in range(n_erode_fat):
            preserved = preserve_area == sli
            sli = morphology.binary_erosion(sli).astype(int)
            sli[preserved == 1] = 1
        labeled = measure.label(sli)
        #sys.exit()
        
        try:
            props = measure.regionprops_table(labeled, properties={'label','area','centroid'})
            
            #max_area = max(props['area'])
            #max_index = list(props['area']).index(max_area)
            #max_label = props['label'][max_index]
            
            keep_labels, keep_areas, keep_centroids, dists_to_center = [], [], [], []
            
            for label, area, c1, c2 in zip(props['label'], props['area'], props['centroid-0'],props['centroid-1']):
                if area > min_shape_area:
                    keep_labels.append(label)
                    keep_areas.append(area)
                    keep_centroids.append((c1,c2))
                    
                    dist = abs( ( (c1-center_x)**2 + (c2-center_y)**2 )**0.5 )
                    
                    dists_to_center.append(dist)
                    
            min_dist = min(dists_to_center)
            min_index = dists_to_center.index(min_dist)
            min_label = keep_labels[min_index]
            
            sli[labeled != min_label] = 0
            for j in range(n_erode_fat):
                sli = morphology.binary_dilation(sli)
                
            
            sli[arm_slice == 1] = 0
            
            if n_erode_fat > 0:
                sli = morphology.binary_closing(sli)
            
            fat_mask[:,:,i] = sli
        except (IndexError, ValueError):
            fat_mask[:,:,i] = zeroes_mask[:,:,i] # happens when there are (no regions, no regions area > min_shape_area)
            
    
    # get the edges of the fat mask, dilate it on each slice and then compare overlap to get probability map of arm edge
    n_dil = 7
    intersection_mask = np.zeros_like(edge_mask)
    for i in range(0, edge_mask.shape[2]):
        sli = fat_mask[:,:,i].copy() + water_mask[:,:,i].copy()
        dil_fat = morphology.binary_dilation(sli)
        fat_ring = (dil_fat.astype(int) - sli.astype(int)).astype(bool)
        
        for j in range(n_dil):
           fat_ring = morphology.binary_dilation(fat_ring)
        
        fat_ring[sli >= 1] = 0
        
        intersection_mask[:,:,i] = fat_ring.copy()
        
    flat_intersect = np.mean(intersection_mask, axis=2)
    
    intermediates['flat_intersect'] = flat_intersect
    
    flat_intersect_nozero = flat_intersect.copy()
    flat_intersect_nozero[flat_intersect_nozero == 0] = np.nan
    
    prop_mask = flat_intersect.copy()
    #thresh_val = np.nanpercentile(flat_intersect_nozero, 10)
    prop_mask[prop_mask < prop_thresh] = 0
    prop_mask[prop_mask >= prop_thresh] = 1
    #prop_mask[prop_mask < thresh_val] = 0
    #prop_mask[prop_mask >= thresh_val] = 1
    
    skeleton = morphology.skeletonize(prop_mask)
    
    # now eliminate shapes with high solidity - arm edges will be very empty shapes
    skeleton_stripped = skeleton.copy()
    skeleton_labeled = morphology.label(skeleton_stripped)
    
    props = measure.regionprops_table(skeleton_labeled, properties={'label','solidity'})
    
    keep_labels, keep_solidities = [], []
    
    for label, solidity in zip(props['label'], props['solidity']):
        if solidity < solidity_thresh:
            keep_labels.append(label)
            keep_solidities.append(solidity)
        else:
            skeleton_stripped[skeleton_labeled == label] = 0
            
    # finally, remove dangling bits - pixels that don't have at least two connected pixels
    skeleton_undangled = skeleton_stripped.copy()
    converged = False
    while not converged:
        altered_pixel = False
        
        skeleton_undangled_connectivity = np.zeros_like(skeleton_undangled)
        for ix in range(skeleton_undangled.shape[0]):
            if ix in [0, skeleton_undangled.shape[0]-1]: # ignore perimeter
                #skeleton_undangled_connectivity[ix,:] = 3
                continue
            for iy in range(skeleton_undangled.shape[1]):
                if iy in [0, skeleton_undangled.shape[1]-1]: # ignore perimeter
                    #skeleton_undangled_connectivity[:,iy] = 3
                    continue
                
                center_val = skeleton_undangled[ix,iy]
                if center_val == False: # don't care if we're centered on a non-skeleton pixel
                    continue
                
                window_sum = skeleton_undangled[ix-1:ix+2,iy-1:iy+2].sum()
                skeleton_undangled_connectivity[ix,iy] = window_sum
                
                if window_sum in [1,2]: #con 1 is isolated which should never happen anyway. con 2 means we're at the end of a dangler
                    altered_pixel = True
                    skeleton_undangled[ix,iy] = False
                    
        converged = not altered_pixel
    
    combined_mask = np.zeros_like(fat_mask)
    combined_mask = combined_mask + fat_mask
    combined_mask = combined_mask + water_mask*2
    # end result: 0 = nothing, 1 = fat, 2 = muscle, 3 = overlap
    
    
    intermediates['edge_mask'] = edge_mask
    intermediates['hole_mask'] = hole_mask
                            
    #skel_cmap = copy.copy(plt.cm.get_cmap('Reds'))
    #skel_cmap.set_bad(alpha=0)
    skel_display = skeleton_stripped.copy().astype(float)
    skel_display[skel_display == 0] = np.nan
    
    skel_undang_display = skeleton_undangled.copy().astype(float)
    skel_undang_display[skel_undang_display == 0] = np.nan
    
    m = np.ma.masked_where(np.isnan(skel_display),skel_display)
    m0 = np.ma.masked_where(np.isnan(skel_undang_display),skel_undang_display)
    
    intermediates['raw_skeleton'] = m
    intermediates['skeleton'] = m0
    
    
    # now use the skeleton to once again punch out the centermost shape in each slice
    n_erode_all = 0
    tissues_masks = [water_mask, fat_mask]
    for tissue_mask in tissues_masks:
        for i in range(bulk_mask.shape[2]):
            sli = tissue_mask[:,:,i]
            
            sli[skeleton_undangled == 1] = 0
            
            # erode n times, keep shapes with area > min_shape_area, select shape with centroid closest to center, dilate n times
            for j in range(n_erode_all):
                sli = morphology.binary_erosion(sli)
            labeled = measure.label(sli, connectivity=1) #limited connectivity
            
            try:
                props = measure.regionprops_table(labeled, properties={'label','area','centroid'})
                
                #max_area = max(props['area'])
                #max_index = list(props['area']).index(max_area)
                #max_label = props['label'][max_index]
                
                keep_labels, keep_areas, keep_centroids, dists_to_center = [], [], [], []
                
                for label, area, c1, c2 in zip(props['label'], props['area'], props['centroid-0'],props['centroid-1']):
                    if area > min_shape_area:
                        keep_labels.append(label)
                        keep_areas.append(area)
                        keep_centroids.append((c1,c2))
                        
                        dist = abs( ( (c1-center_x)**2 + (c2-center_y)**2 )**0.5 )
                        
                        dists_to_center.append(dist)
                        
                min_dist = min(dists_to_center)
                min_index = dists_to_center.index(min_dist)
                min_label = keep_labels[min_index]
                
                sli[labeled != min_label] = 0
                for j in range(n_erode_all):
                    sli = morphology.binary_dilation(sli)
                    
                
                tissue_mask[:,:,i] = sli
            except (IndexError, ValueError):
                water_mask[:,:,i] = zeroes_mask[:,:,i].copy() # happens when there are (no regions, no regions area > min_shape_area)
    
    # update the combined mask
    combined_mask = np.zeros_like(fat_mask)
    combined_mask = combined_mask + fat_mask
    combined_mask = combined_mask + water_mask*2
    
    
    return water_mask, fat_mask, combined_mask, intermediates



def restore_coordspace_from_source(nif, source, out_nif=None, bin_folder=None, path_to_dcm2nii='/Users/skyjones/Desktop/ASE_standalone_v1/slw_resources/dcm2nii64'):
    """
    Takes a nifti that lost its affine and heading and associates the affine/heading from a source image
    
    
    """
    
    
    in_nif = nib.load(nif)
    
    try:
        in_source = nib.load(source)
    except nib.filebasedimages.ImageFileError: # when the source is an enhanced dicom
        warnings.warn('DCM source image. Attempting conversion to NiFTI for coordinate space extraction')
        if bin_folder is not None:
            working_folder = bin_folder
        else:
            working_folder = os.path.join(os.path.dirname(nif), 'coordinate_working')
        
        os.mkdir(working_folder)
        conversion_command = f'{path_to_dcm2nii} -a n -i n -d n -p n -e n -f y -v n -o {working_folder} {source}'
        os.system(conversion_command)
        
        converted_file = os.path.join(working_folder, os.path.basename(source.replace('DCM', 'nii.gz')))
        in_source = nib.load(converted_file)
        
        if bin_folder is None:
            shutil.rmtree(working_folder)
        
    
    restored_affine = in_source.affine
    restored_header = in_source.header
    
    restored_data = np.flip(in_nif.get_fdata(), axis=1)
    
    out_image = nib.Nifti1Image(restored_data, restored_affine, restored_header)
    
    if out_nif is not None:
        out_name = out_nif
    else:
        out_name = nif
    
    nib.save(out_image, out_name)
    

def dcm_to_nii(in_dicom, path_to_dcm2nii='/Users/skyjones/Desktop/ASE_standalone_v1/slw_resources/dcm2nii64'):
    """
    does not work correctly with 4d files
    """
    
    conversion_command = f'{path_to_dcm2nii} -a n -i n -d n -p n -e n -f y -v n {in_dicom}'
    os.system(conversion_command)
    
    
    
    
    