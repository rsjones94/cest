#!/usr/bin/env python3

import os
import glob
import shutil
import datetime
import operator
import sys
import copy

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


mpl.rcParams.update({'errorbar.capsize': 2})

# input
master_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/'
cest_spreadsheet = '/Users/skyjones/Desktop/hiv_processing/pos_workbook.xlsx'

#output
data_out = '/Users/skyjones/Desktop/hiv_processing/data/working/results.csv'
figure_folder = '/Users/skyjones/Desktop/hiv_processing/data/working/figures/'

do_prep = True
do_figs = True



##########

weight_types = ['fat', 'water']

if do_prep:
    cest_df = pd.read_excel(cest_spreadsheet)
    #cest_df['mri_id'] = cest_df['mri_id'].astype(str)
    
    subs = glob.glob(os.path.join(master_folder, '*/'))
    subs = [sub for sub in subs if 'aim' in sub.lower()]
    
    for sub in subs:
        aim_folder = sub[:-1].split('/')[-1]
        
        if aim_folder == 'aim1':
            continue
        
        aim_figure_folder = os.path.join(figure_folder, aim_folder)
        if os.path.exists(aim_figure_folder):
            shutil.rmtree(aim_figure_folder)
        os.mkdir(aim_figure_folder)
        
        processed_folder = os.path.join(sub, 'processed')
        data_folders = glob.glob(os.path.join(processed_folder, '*/'))
        
        for fol in data_folders:
            base = os.path.basename(fol[:-1])
            splitter = base.split('_')
            cest_id = '_'.join(splitter[:-2])
            num_id = '_'.join(splitter[1:-2])
            arm = splitter[-1]
            
            scan_id = '_'.join(splitter[:2])
            
            # get the rows in the cest_df that correspond to this mri_id
            
            #the_rows = cest_df[cest_df['mri_id'] == num_id]
            the_rows = cest_df[cest_df['scan_id'] == scan_id]
            
            #if num_id != '146291':
            #    continue
            
            print(f'{aim_folder}: {num_id}, {arm}')
                        
            
            
            water_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C1_.nii.gz')
            fw_inphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C2_.nii.gz')
            fw_outphase = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C3_.nii.gz')
            fat_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C4_.nii.gz')
            t2star_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C5_.nii.gz')
            #b0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C6_.nii.gz') # this is actuallu just the water-weighted b0
            
            analysis_folder = os.path.join(fol, 'analysis')
            if os.path.exists(analysis_folder):
                shutil.rmtree(analysis_folder)
            #os.mkdir(analysis_folder)
            
            id_figure_folder = os.path.join(aim_figure_folder, num_id)
            if not os.path.exists(id_figure_folder):
                os.mkdir(id_figure_folder)
            
            
            fat_image_loaded = nib.load(fat_image)
            fat_image_data = fat_image_loaded.get_fdata()
            fat_image_data = np.mean(fat_image_data, 3) # time average
            fat_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            water_image_loaded = nib.load(water_image)
            water_image_data = water_image_loaded.get_fdata()
            water_image_data = np.mean(water_image_data, 3) # time average
            water_image_voxelvol = np.product(fat_image_loaded.header['pixdim'][1:4])
            
            t2star_image_loaded = nib.load(t2star_image)
            t2star_image_data = t2star_image_loaded.get_fdata()
            t2star_image_data = np.mean(t2star_image_data, 3) # time average
            
                                
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 12))
            
            for axrow in axs:
                for ax in axrow:
                    ax.axis('off')
            
            # fat
            
            fat_thresh = threshold_otsu(fat_image_data)
            fat_mask = fat_image_data > fat_thresh
            #fat_mask = morphology.closing(fat_mask)
            
            fat_slice = fat_image_data[:,:,5]
            fat_mask_slice = fat_mask[:,:,5]
            
            axs[0][0].imshow(fat_slice, cmap='gray')
            axs[0][0].set_title('fat-weighted')
            axs[0][1].imshow(fat_mask_slice, cmap='binary')
            
            # muscle
    
            water_thresh = threshold_otsu(water_image_data)
            water_mask = water_image_data > water_thresh
            #water_mask = morphology.closing(water_mask)
    
            
            water_slice = water_image_data[:,:,5]
            water_mask_slice = water_mask[:,:,5]
            
            axs[1][0].imshow(water_slice, cmap='gray')
            axs[1][0].set_title('water-weighted')
            axs[1][1].imshow(water_mask_slice, cmap='binary')
            
                
            # get fat edges
            high_ratio = 0.05
            low_ratio = 0.02
            edge_mask = np.zeros_like(fat_image_data)
            for i in range(edge_mask.shape[2]):
                sli = fat_image_data[:,:,i]
                
                wm = water_mask[:,:,i]
                fm = fat_mask[:,:,i]
                
                #wm_dilated = morphology.binary_dilation(wm).astype(int) # muscles dilated once
                #fm_dilated = morphology.binary_dilation(morphology.binary_dilation(fm)).astype(int) # fat dilated twice
                
                high_thresh = np.percentile(sli, 95) * high_ratio
                low_thresh = high_thresh * low_ratio
                
                edges = feature.canny(sli, sigma=2)
                edge_mask[:,:,i] = edges
                
            
            # overlay
            
            # first, fat beats water
            water_mask[fat_mask > 0] = 0
            water_mask_slice = water_mask[:,:,5]
            
            
            combined_mask = np.zeros_like(fat_mask)
            
            combined_mask = combined_mask + fat_mask
            combined_mask = combined_mask + water_mask*2
            # end result: 0 = nothing, 1 = fat, 2 = muscle, 3 = overlap
            
            fat_slice = fat_image_data[:,:,5]
            combined_mask_slice = combined_mask[:,:,5]
    
    
            t2star_slice = t2star_image_data[:,:,5]
            
            # custom cmap
            cmap = plt.cm.jet  # define the colormap
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            # force the first color entry to be grey
            cmaplist[0] = (.5, .5, .5, 1.0)
            # create the new map
            cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            # define the bins and normalize
            bounds = np.linspace(0, 4, 5)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        
            
            
            #axs[2][0].imshow(t2star_slice, cmap='gray')
            #axs[2][0].set_title('t2*')
            
            #ls = LightSource(azdeg=315, altdeg=45)
            #axs[2][0].imshow(ls.hillshade(fat_slice, vert_exag=1, dx=1, dy=1), cmap='gray')
            #axs[2][0].set_title('Fat-weighted hillshade')
            
            dy,dx = np.gradient(fat_slice)
            theta = np.degrees(np.arctan2(dy,dx))
            axs[2][0].imshow(theta, cmap='twilight_shifted', vmin=-180, vmax=180, interpolation='nearest')
            axs[2][0].set_title('Fat-weighted directions')
            
            hillshade_image_name = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_fat_hillshade_{arm}')
    
            axs[2][1].imshow(combined_mask_slice, cmap=cmap, norm=norm, interpolation='nearest')
            axs[2][1].set_title('combined mask')
            
       
            bulk_mask = combined_mask > 0
            center_x = bulk_mask.shape[0] / 2
            center_y = bulk_mask.shape[1] / 2
            zeroes_mask = np.zeros_like(bulk_mask).astype(bool)
            
            # find the arm muscle
            n_erode_muscle = 1
            for i in range(bulk_mask.shape[2]):
                sli = water_mask[:,:,i]
                
                # erode n times, keep shapes with area > 100, select shape with centroid closest to center, dilate n times
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
                        if area > 100:
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
                        
                    sli = morphology.binary_closing(sli)
                    
                    water_mask[:,:,i] = sli
                except (IndexError, ValueError):
                    water_mask[:,:,i] = zeroes_mask[:,:,i].copy() # happens when there are (no regions, no regions area > 100)
                    
            
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
            
            '''
            # then dilate each label in the edge mask three times in each slice and use the overlap to guess where the arm-torso interface is and propagate that
            n_dil = 10
            intersection_mask = np.zeros_like(edge_mask)
            for i in range(0, edge_mask.shape[2]):
                sli = edge_mask[:,:,i].copy()
                intersection_counter = np.zeros_like(sli)
                ra = list(np.unique(sli))
                removers = [-1,0]
                for m in removers:
                    try:
                        ra.pop(ra.index(m))
                    except ValueError:
                        continue
                for j in ra: # don't care about -1, that represents filtered edges
                    binaried = sli == j
                    for m in range(0,n_dil):
                        binaried = morphology.binary_dilation(binaried)
                    intersection_counter = intersection_counter + binaried
                    
                intersection_bin = intersection_counter.copy()
                intersection_bin[intersection_bin < 2] = 0 # basically we summed up all the dilated shapes. >1 means there was an intersection
                intersection_bin[intersection_bin >= 2] = 1 # basically we summed up all the dilated shapes. >1 means there was an intersection
                
                intersection_bin[fat_mask[:,:,i] == 1] = 0 # blot out the fat mask on that slice though
                
                intersection_mask[:,:,i] = intersection_bin
                
                #plt.figure()
                #plt.imshow(intersection_bin)
                
            flat_intersect = np.mean(intersection_mask, axis=2)
            
            prop_mask = flat_intersect.copy()
            prop_mask[prop_mask < 0.25] = 0
            prop_mask[prop_mask >= 0.25] = 1
            '''
            
            
            # find the arm fat
            n_fat_preserve = 2 # number of dilations around arm muscle to preserve fat
            n_erode_fat = 1
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
                # erode n times, keep shapes with area > 100, select shape with centroid closest to center, dilate n times
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
                        if area > 100:
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
                    sli = morphology.binary_closing(sli)
                    
                    fat_mask[:,:,i] = sli
                except (IndexError, ValueError):
                    fat_mask[:,:,i] = zeroes_mask[:,:,i] # happens when there are (no regions, no regions area > 100)
                    
            
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
            flat_intersect_nozero = flat_intersect.copy()
            flat_intersect_nozero[flat_intersect_nozero == 0] = np.nan
            
            prop_mask = flat_intersect.copy()
            #thresh_val = np.nanpercentile(flat_intersect_nozero, 10)
            prop_mask[prop_mask < 0.2] = 0
            prop_mask[prop_mask >= 0.2] = 1
            #prop_mask[prop_mask < thresh_val] = 0
            #prop_mask[prop_mask >= thresh_val] = 1
            
            skeleton = morphology.skeletonize(prop_mask)
            
            # now eliminate shapes with high solidity - arm edges will be very empty shapes
            skeleton_stripped = skeleton.copy()
            skeleton_labeled = morphology.label(skeleton_stripped)
            
            props = measure.regionprops_table(skeleton_labeled, properties={'label','solidity'})
            
            keep_labels, keep_solidities = [], []
            
            for label, solidity in zip(props['label'], props['solidity']):
                if solidity < 0.1:
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
                            
             
            '''
            plt.figure()
            plt.imshow(flat_intersect)
            
            plt.figure()
            plt.imshow(prop_mask)     
            
            plt.figure()
            plt.imshow(skeleton)
            
            plt.figure()
            plt.imshow(skeleton_stripped)
            '''
                
            fat_mask_slice = fat_mask[:,:,5]
            water_mask_slice = water_mask[:,:,5]
            
            combined_mask = np.zeros_like(fat_mask)
            combined_mask = combined_mask + fat_mask
            combined_mask = combined_mask + water_mask*2
            # end result: 0 = nothing, 1 = fat, 2 = muscle, 3 = overlap
            combined_mask_slice = combined_mask[:,:,5]
            
            #new_cmap = copy.copy(plt.cm.get_cmap('Reds'))
            #new_cmap.set_bad(alpha=0)
            
            edge_slice = edge_mask[:,:,5]
            hole_slice = hole_mask[:,:,5]
            
            #print(f'\t{edging_slice.sum()}')
            #axs[3][0].imshow(edging_slice, cmap='YlGn')
            axs[3][0].imshow(edge_slice, cmap='jet', interpolation='nearest')
            axs[3][0].set_title('fat edges')
            
            axs[3][1].imshow(combined_mask_slice, cmap=cmap, norm=norm, interpolation='nearest')
            axs[3][1].imshow(hole_slice, cmap='Reds', interpolation='nearest', alpha=0.5)
            axs[3][1].set_title('arm mask')
            
                                    
            #skel_cmap = copy.copy(plt.cm.get_cmap('Reds'))
            #skel_cmap.set_bad(alpha=0)
            skel_display = skeleton_stripped.copy().astype(float)
            skel_display[skel_display == 0] = np.nan
            
            skel_undang_display = skeleton_undangled.copy().astype(float)
            skel_undang_display[skel_undang_display == 0] = np.nan
            
            m = np.ma.masked_where(np.isnan(skel_display),skel_display)
            m0 = np.ma.masked_where(np.isnan(skel_undang_display),skel_undang_display)
            axs[3][1].imshow(m, cmap='Reds', vmin=0, vmax=2, interpolation='nearest')
            axs[3][1].imshow(m0, cmap='Greens', vmin=0, vmax=2, interpolation='nearest')
            
            
            
            # now use the skeleton to once again punch out the centermost shape in each slice
            n_erode_all = 0
            tissues_masks = [water_mask, fat_mask]
            for tissue_mask in tissues_masks:
                for i in range(bulk_mask.shape[2]):
                    sli = tissue_mask[:,:,i]
                    
                    sli[skeleton_undangled == 1] = 0
                    
                    # erode n times, keep shapes with area > 100, select shape with centroid closest to center, dilate n times
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
                            if area > 100:
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
                        water_mask[:,:,i] = zeroes_mask[:,:,i].copy() # happens when there are (no regions, no regions area > 100)
            
            # update the combined mask
            combined_mask = np.zeros_like(fat_mask)
            combined_mask = combined_mask + fat_mask
            combined_mask = combined_mask + water_mask*2
            combined_mask_slice = combined_mask[:,:,5]
            
    
            my_cmap = copy.copy(plt.cm.get_cmap('inferno'))
            my_cmap.set_bad(alpha=0)
            
            #norm = mpl.colors.BoundaryNorm((combined_mask_slice.min(), combined_mask_slice.max()), my_cmap.N)
            
            '''
            apt_slice = apt_image_data[:,:,5]
            axs[4][0].imshow(apt_slice, cmap='inferno', interpolation='nearest')
            axs[4][0].set_title('APT')
            '''
            
            """
            apt_slice_arm = apt_slice.copy()
            apt_slice_arm[combined_mask_slice == 0] = np.nan
            axs[4][1].imshow(apt_slice_arm, cmap=my_cmap, interpolation='nearest')
            axs[4][1].set_title('APT (arm)')
            """
            axs[4][0].imshow(flat_intersect, cmap='Reds', interpolation='nearest')
            axs[4][0].set_title('Flattened edges')
            
            
            axs[4][1].imshow(fat_slice, cmap='gray')
            axs[4][1].imshow(combined_mask_slice, cmap=cmap, norm=norm, interpolation='nearest', alpha=0.5)
            axs[4][1].set_title('Refined mask')
    
            inspect_image_name = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_roi_{arm}')
            
            
            main_title = ''
            
            # save the muscle and fat masks, using the water image affine and header
            water_affine = water_image_loaded.affine
            water_header = water_image_loaded.header
            
            muscle_nifti = nib.Nifti1Image(water_mask, water_affine, water_header)
            muscle_nifti_path =  os.path.join(fol, f'{cest_id}_cestdixon_{arm}_musclemask.nii.gz')
            nib.save(muscle_nifti, muscle_nifti_path)
            
            fat_nifti = nib.Nifti1Image(fat_mask, water_affine, water_header)
            fat_nifti_path =  os.path.join(fol, f'{cest_id}_cestdixon_{arm}_fatmask.nii.gz')
            nib.save(fat_nifti, fat_nifti_path)
            
                        
            # now check to see if a manual mask already exists. if it does, use that instead of the automated mask
            manual_muscle_file = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_musclemask_manual.nii.gz')
            if os.path.exists(manual_muscle_file):
                print('\tUsing manual muscle mask!!!!!!!!!!')
                manual_muscle_loaded = nib.load(manual_muscle_file)
                manual_muscle_data = manual_muscle_loaded.get_fdata()
                water_mask = manual_muscle_data
                
                main_title = main_title + 'Muscle mask: manual ; '
                axs[1][1].set_title('water mask (overridden)')
            else:
                print('\tUsing automated muscle mask')
                main_title = main_title + 'Muscle mask: automatic ; '
                axs[1][1].set_title('water mask (not overridden)')
                
            manual_fat_file = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_fatmask_manual.nii.gz')
            if os.path.exists(manual_fat_file):
                print('\tUsing manual fat mask!!!!!!!!!!')
                manual_fat_loaded = nib.load(manual_fat_file)
                manual_fat_data = manual_fat_loaded.get_fdata()
                fat_mask = manual_fat_data
                
                main_title = main_title + 'Fat mask: manual'
                axs[0][1].set_title('fat mask (overridden)')
            else:
                print('\tUsing automated fat mask')
                main_title = main_title + 'Fat mask: automatic'
                axs[0][1].set_title('fat mask (not overridden)')
                
            
            
            #plt.suptitle(main_title)
            plt.tight_layout()
            plt.savefig(inspect_image_name)
            plt.close(fig)
            
            
            fat_vol = fat_mask.sum() * fat_image_voxelvol # in mm**3
            muscle_vol = water_mask.sum() * water_image_voxelvol # in mm**3
            fat_frac = (fat_vol) / (muscle_vol)
            
            for idx in the_rows.index:
                cest_df.at[idx, f'{arm}_fat_vol_mm3'] = fat_vol
                cest_df.at[idx, f'{arm}_muscle_vol_mm3'] = muscle_vol
                cest_df.at[idx, f'{arm}_fat_frac'] = fat_frac
            

            for weight_type in weight_types:
                print(f'\tWeighting: {weight_type}')

                
                apt_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_APT_{weight_type}.nii.gz')
                noe_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_OPPAPT_{weight_type}.nii.gz')
                zspec_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_ZSPECTRUM_{weight_type}.nii.gz')
                s0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_S0_{weight_type}.nii.gz')
                uncorrected_zspec_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}OCEST_uncorrected_{weight_type}.nii.gz')
                
                apt_image_loaded = nib.load(apt_image)
                apt_image_data = apt_image_loaded.get_fdata()
                
                noe_image_loaded = nib.load(noe_image)
                noe_image_data = noe_image_loaded.get_fdata()
                
                zspec_image_loaded = nib.load(zspec_image)
                zspec_image_data = zspec_image_loaded.get_fdata()
                zspec_image_data[np.isclose(zspec_image_data,0)] = np.nan
                
                uncorrected_zspec_image_loaded = nib.load(uncorrected_zspec_image)
                uncorrected_zspec_image_data = uncorrected_zspec_image_loaded.get_fdata()
                #note that we don't need to set 0 pixels to nan in the uncorrected as the uncorrected image gives signal everywhere
                
                s0_image_loaded = nib.load(s0_image)
                s0_image_data = s0_image_loaded.get_fdata()
                
                s0_image_data = np.where(np.isclose(s0_image_data,0), np.nan, s0_image_data)
                
                zspec_image_data_norm = zspec_image_data.copy()
                for i in range(zspec_image_data_norm.shape[-1]):
                    zspec_image_data_norm[:,:,:,i] = zspec_image_data_norm[:,:,:,i] / s0_image_data
                
                apt_image_data_norm = apt_image_data / s0_image_data
                
                noe_image_data_norm = noe_image_data / s0_image_data
        

                    
                '''   
                # now calculate the normalization coefficient, i.e., the mean intensity of the first three and last three chemical shifts in the tissue
                #dual_roi = (water_mask | fat_mask)
                zspec_copy_fat = zspec_image_data.copy()
                zspec_copy_water = zspec_image_data.copy()
                
                for i in range(zspec_copy_fat.shape[-1]):
                    zspec_copy_fat[:,:,:,i][fat_mask != 1] = np.nan # mask it
                    zspec_copy_water[:,:,:,i][water_mask != 1] = np.nan # mask it
                
                first_three_fat = np.nanmean(zspec_copy_fat[:,:,:,:3])
                last_three_fat = np.nanmean(zspec_copy_fat[:,:,:,-3:])
                norm_coef_fat = np.mean([first_three_fat, last_three_fat])
                
                first_three_water = np.nanmean(zspec_copy_water[:,:,:,:3])
                last_three_water = np.nanmean(zspec_copy_water[:,:,:,-3:])
                norm_coef_water = np.mean([first_three_water, last_three_water])
                '''
                
                
                rois = [water_mask, fat_mask]
                roi_names = ['muscle', 'fat']
                #norm_coefs = [norm_coef_water, norm_coef_fat]
                # iterate through the rois to calculate signal stuff
                sig_cmaps = ['inferno', 'inferno']
                sigs = [apt_image_data_norm, noe_image_data_norm]
                sig_names = ['apt', 'noe']
        
                for roi, roi_n in zip(rois, roi_names):
                    for sig, sig_name in zip(sigs, sig_names):
                        sig_copy = sig.copy()
                        sig_copy[roi != 1] = np.nan # mask it
                        #sig_copy[sig_copy == 0] = np.nan # remove zero signal wierdery
                        
                        the_mean = np.nanmean(sig_copy)
                        the_std = np.nanstd(sig_copy)
                        the_95 = np.nanpercentile(sig_copy, 95)
                        the_5 = np.nanpercentile(sig_copy, 5)
                        
                        for idx in the_rows.index:
                            cest_df.at[idx, f'{arm}_{roi_n}_mean_{sig_name}_{weight_type}weighted'] = the_mean
                            cest_df.at[idx, f'{arm}_{roi_n}_std_{sig_name}_{weight_type}weighted'] = the_std
                            cest_df.at[idx, f'{arm}_{roi_n}_95th_{sig_name}_{weight_type}weighted'] = the_95
                            cest_df.at[idx, f'{arm}_{roi_n}_5th_{sig_name}_{weight_type}weighted'] = the_5
                            
                # now make multislice figures
                muscle_col = 'Greens'
                fat_col = 'Blues'
                clims = [(-1,2),(-1,2)]
                for sig, sig_name, sig_cmap, clim in zip(sigs, sig_names, sig_cmaps, clims):
                    the_index = 0
                    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 12))
                    for ax_row in axs:
                        for ax in ax_row:
                            ax.axis('off')
                            ax.set_title(f'Slice {the_index}: {sig_name}, {weight_type} weighted')
                            muscle_mask_slice = water_mask[:,:,the_index]
                            fat__mask_slice = fat_mask[:,:,the_index]
                            sig_slice = sig[:,:,the_index]
                            sig_slice[sig_slice == 0] = np.nan
                            
                            muscle_mask_slice = np.nan_to_num(muscle_mask_slice)
                            fat_mask_slice = np.nan_to_num(fat_mask_slice)
                            
                            muscle_expand = morphology.binary_dilation(muscle_mask_slice).astype(int)
                            fat_expand = morphology.binary_dilation(fat_mask_slice).astype(int)
                            
                            muscle_ring = (muscle_expand - muscle_mask_slice).astype(float)
                            fat_ring = (fat_expand - fat_mask_slice).astype(float)
                            
                            muscle_ring[muscle_ring == 0] = np.nan
                            fat_ring[fat_ring == 0] = np.nan
                            
                            muscle_cmap = copy.copy(plt.cm.get_cmap(muscle_col))
                            muscle_cmap.set_bad(alpha=0)
                            
                            fat_cmap = copy.copy(plt.cm.get_cmap(fat_col))
                            fat_cmap.set_bad(alpha=0)
                            
                            sig_cmap_alt = copy.copy(plt.cm.get_cmap(sig_cmap))
                            sig_cmap_alt.set_bad(alpha=0)
                            
                            show = ax.imshow(sig_slice, cmap=sig_cmap_alt, interpolation='nearest', alpha=0.7, vmin=clim[0], vmax=clim[1])
                            ax.imshow(muscle_ring, cmap=muscle_cmap, vmin=0, vmax=2, interpolation='nearest')
                            ax.imshow(fat_ring, cmap=fat_cmap, vmin=0, vmax=2, interpolation='nearest')
                            
                            cbar = plt.colorbar(show, ax=ax)
                            
                            the_index += 1
                
                    multislice_image_name = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_multislice_{sig_name}_{arm}_{weight_type}weighted')
                    
                    plt.tight_layout()
                    plt.savefig(multislice_image_name)
                    plt.close()
                    
                    
                # time to parse the zspec data
                
                #central_index = 55 # chem shift = 0ppm here. note in matlab code this index is 56, but python is 0-indexed
                # actually I think the central index is literally just the middle index
                central_index = int(np.ceil(zspec_image_data.shape[3] / 2)) - 1
                shift_step = 0.1 # change in chemical shift between each index
                n_steps = zspec_image_data.shape[3]
                
                start_shift = -(central_index * shift_step)
                end_shift = (n_steps - central_index) * shift_step
                
                shifts = np.arange(start_shift, end_shift - shift_step, shift_step)
                shifts = [round(i,1) for i in shifts] # round to nearest tenth. thx floating point imprecision
                
                # also set up the uncorrected spectrum
                central_index_uncor = int(np.ceil(uncorrected_zspec_image_data.shape[3] / 2)) - 1
                shift_step_uncor = 0.1 # change in chemical shift between each index
                n_steps_uncor = uncorrected_zspec_image_data.shape[3]
                
                start_shift_uncor = -(central_index_uncor* shift_step_uncor)
                end_shift_uncor = (n_steps_uncor - central_index_uncor) * shift_step_uncor
                
                shifts_uncor = np.arange(start_shift_uncor, end_shift_uncor - shift_step_uncor, shift_step_uncor)
                shifts_uncor = [round(i,1) for i in shifts_uncor] # round to nearest tenth. thx floating point imprecision
                
                for idx in the_rows.index:
                    cest_df.at[idx, f'chemical_shifts_ppm'] = str(shifts)
                    cest_df.at[idx, f'chemical_shifts_uncor_ppm'] = str(shifts_uncor)
                
                bulk_means = []
                bulk_normalized_means = []
                bulk_stds = []
                bulk_normalized_stds = []
                bulk_asyms = []
                bulk_difs = []
                bulk_means_uncor = []
                bulk_stds_uncor = []
                for roi, roi_n in zip(rois, roi_names):
                    
                    # calculate the zspec
                    means = []
                    stds = []
                    for i, shift in enumerate(shifts):
                        boxed = zspec_image_data[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[roi != 1] = np.nan # mask it
                        
                        the_mean = np.nanmean(boxed_copy)
                        the_std = np.nanstd(boxed_copy)
                        
                        means.append(the_mean)
                        stds.append(the_std)
                        
                    # now normalize the means
                    means_norm = []
                    stds_norm = []
                    for i, shift in enumerate(shifts):
                        boxed = zspec_image_data_norm[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[roi != 1] = np.nan # mask it
                        #boxed_copy = boxed_copy / norm_coef
                        
                        the_mean_norm = np.nanmean(boxed_copy)
                        the_std_norm = np.nanstd(boxed_copy)
                        
                        means_norm.append(the_mean_norm)
                        stds_norm.append(the_std_norm)
                        
                    # now get the uncorrected means
                    means_uncor = []
                    stds_uncor = []
                    for i, shift in enumerate(shifts_uncor):
                        boxed = uncorrected_zspec_image_data[:,:,:,i]
                        boxed_copy = boxed.copy()
                        boxed_copy[roi != 1] = np.nan # mask it
                        #boxed_copy = boxed_copy / norm_coef
                        
                        the_mean_uncor = np.nanmean(boxed_copy)
                        the_std_uncor = np.nanstd(boxed_copy)
                        
                        means_uncor.append(the_mean_uncor)
                        stds_uncor.append(the_std_uncor)
                        
                    # now calculate the asymmetry
                    # f(x) = [v(-x) - v(x)] / x
                    asym = []
                    for i, (shift, mean) in enumerate(zip(shifts, means)):
                        try:
                            neg_i = shifts.index(-shift)
                        except ValueError: # occurs when there is no symmetric shift val
                            asym.append(np.nan)
                            continue
                            
                        
                        asym_mean = means[neg_i]
                        try:
                            val = (mean - asym_mean) / shift
                        except ZeroDivisionError:
                            asym.append(np.nan)
                            continue
                            
                        asym.append(val)
                        
                    # now calculate the equilibrium difference
                    # f(x) = v_0 - v(x)
                    dif = []
                    for i, (shift, mean) in enumerate(zip(shifts, means)):
                        
                        dif_val = mean - means[0]
                        dif.append(dif_val)
                        
                    
    
                    
                    for idx in the_rows.index:
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_mean_{weight_type}weighted'] = str(means)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_std_{weight_type}weighted'] = str(stds)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_asym_{weight_type}weighted'] = str(asym)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_eqdif_{weight_type}weighted'] = str(dif)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_meannorm_{weight_type}weighted'] = str(means_norm)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_stdnorm_{weight_type}weighted'] = str(stds_norm)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_mean_uncor_{weight_type}weighted'] = str(means_uncor)
                        cest_df.at[idx, f'{roi_n}_{arm}_signals_std_uncor_{weight_type}weighted'] = str(stds_uncor)
                    
                    bulk_means.append(means)
                    bulk_stds.append(stds)
                    bulk_asyms.append(asym)
                    bulk_difs.append(dif)
                    bulk_normalized_means.append(means_norm)
                    bulk_normalized_stds.append(stds_norm)
                    bulk_means_uncor.append(means_uncor)
                    bulk_stds_uncor.append(stds_uncor)
                    
    
                    
                    
                        
                    
                    
                figure, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,24))
                zspec_colors = ['blue', 'green']
                for roi, roi_n, means_norm, stds_norm, asyms, difs, c in zip(rois, roi_names, bulk_normalized_means, bulk_normalized_stds, bulk_asyms, bulk_difs, zspec_colors):
                    #ax.errorbar(shifts, means, label=f'{roi_n} +/- 1sd', color=c, yerr=stds, ls='-', marker='.', mec='black', capsize=5)
                    axs[0].fill_between(shifts, np.array(means_norm)+np.array(stds_norm), np.array(means_norm)-np.array(stds_norm), color=c, alpha=0.3)
                    axs[0].plot(shifts, means_norm, label=f'{roi_n} +/- 1sd', color=c)
                    
                    axs[1].plot(shifts, asyms, label=f'{roi_n}', color=c)
                    #axs[2].plot(shifts, difs, label=f'{roi_n}', color=c)
                    
                axs[0].set_xlabel('Chemical shift (ppm)')
                axs[1].set_xlabel('Chemical shift (ppm)')
                #axs[2].set_xlabel('Chemical shift (ppm)')
                
                axs[0].set_ylabel('Normalized signal intensity (a.u.)')
                axs[0].set_title(f'Z Spectrum, {num_id}, {arm}, {weight_type} weighted')
                axs[0].set_ylim(-1,2)
                
                            
                axs[1].set_ylabel('Asymmetry')
                axs[1].set_title(f'Asymmetry')
                            
                #axs[2].set_ylabel('Difference (ppm)')
                #axs[2].set_title(f'Equilibrium difference')
                    
                axs[0].legend()
                axs[1].legend()
                #axs[2].legend()
                plt.tight_layout()
                zspec_figname = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_zspec_{arm}_{weight_type}weighted')
                plt.savefig(zspec_figname)
                
                
                plt.close()
                
            
        
        
                 
        cest_df.to_csv(data_out, index=False)
        

###### figs
if do_figs:
    print('figuring')
    cest_df = pd.read_csv(data_out)
    
    aff_col = 'coral'
    cont_col = 'dodgerblue'
    
    roi_names = ['muscle', 'fat', 'whole']
    sig_names = ['apt', 'noe', 'fat_frac']
    aims = cest_df['aim'].unique()
    for aim in aims:
        
        if aim == 1:
            continue
        
        for weight_type in weight_types:

            target_aim_folder = os.path.join(figure_folder, f'aim{aim}')
            sub_df = cest_df[cest_df['aim'] == aim]
            
            study_ids = sub_df['study_id'].unique()
            
            #zspec comparisons
            zspec_comp_fig_name = os.path.join(target_aim_folder, f'zspec_comp_aim{aim}_{weight_type}weighted.png')
            figure, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
            
            fat_cont_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in sub_df[f'fat_cont_signals_meannorm_{weight_type}weighted'].dropna()])
            fat_cont_signals = np.nanmean(fat_cont_all, 0)
            fat_aff_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in sub_df[f'fat_aff_signals_meannorm_{weight_type}weighted'].dropna()])
            fat_aff_signals = np.nanmean(fat_aff_all, 0)
            
            muscle_cont_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in sub_df[f'muscle_cont_signals_meannorm_{weight_type}weighted'].dropna()])
            muscle_cont_signals = np.nanmean(muscle_cont_all, 0)
            muscle_aff_all = np.array([np.array([float(j) for j in i[1:-1].split(',')]) for i in sub_df[f'muscle_aff_signals_meannorm_{weight_type}weighted'].dropna()])
            muscle_aff_signals = np.nanmean(muscle_aff_all, 0)
            
    
            
            for i, row in sub_df.iterrows():
                shifts = row['chemical_shifts_ppm']
                
    
                
                try:
                    shifts = [float(i) for i in shifts[1:-1].split(', ')]
                except TypeError:
                    continue
                
                
                arms = ['aff', 'cont']
                acols = ['red', 'blue']
                locs = ['muscle', 'fat']
                
                lens_outer = [[len(muscle_aff_all), len(muscle_cont_all)],[len(fat_aff_all), len(fat_cont_all)]]
                
                for ax, loc, lens in zip(axs, locs, lens_outer):
                    for arm, acol, the_len in zip(arms, acols, lens):
                        means = row[f'{loc}_{arm}_signals_meannorm_{weight_type}weighted']
                        stds = row[f'{loc}_{arm}_signals_stdnorm_{weight_type}weighted']
                        
                        try:
                            means = [float(i) for i in means[1:-1].split(', ')]
                            stds = [float(i) for i in stds[1:-1].split(', ')]
                        except TypeError:
                            #print('Type error....')
                            #ebah = means
                            continue
                            
                        
                        ax.fill_between(shifts, np.array(means)+np.array(stds), np.array(means)-np.array(stds), color=acol, alpha=0.3/the_len, lw=0)
                        #ax.plot(shifts, means, color=acol, alpha=0.2)
                        
                    ax.set_title(f'Comparative Z-spectra: {loc} (aim {aim}), {weight_type} weighted')
                    ax.set_xlabel('Chemical shift (ppm)')
                    ax.set_ylabel('Normalized signal intensity (a.u.)')
                    
                    ax.set_ylim(-1,2)
                    
    
            
            alpher = 0.6
            axs[0].plot(shifts, muscle_aff_signals, color=acols[0], label=f'Affected (n={len(muscle_aff_all)})', alpha=alpher)
            axs[0].plot(shifts, muscle_cont_signals, color=acols[1], label=f'Contralateral (n={len(muscle_cont_all)})', alpha=alpher)
            
            axs[1].plot(shifts, fat_aff_signals, color=acols[0], label=f'Affected (n={len(fat_aff_all)})', alpha=alpher)      
            axs[1].plot(shifts, fat_cont_signals, color=acols[1], label=f'Contralateral (n={len(fat_cont_all)})', alpha=alpher)
            
            axs[0].legend()
            axs[1].legend()
                    
            plt.tight_layout()
            plt.savefig(zspec_comp_fig_name, dpi=200)
            
            plt.close()
            
        if aim == 3:
            
                    
            tissue_types = ['muscle', 'fat']
            sides = ['aff', 'cont']
            sig_types = ['apt', 'noe']
            
            
            ## treatment response stats
            
            id_lens = [len(sub_df[sub_df['study_id']==i]) for i in study_ids] # number of treatments. if 4, then we know they have completed pre and post intervention scans for both interventions
            
            #full_ids = [i for i, n in zip(study_ids, id_lens) if n >= 4]
    
    
            response_df = pd.DataFrame()
            response_df['study_id'] = study_ids
            
            
            for tis in tissue_types:
                for sig in sig_types:
                    for weight_type in weight_types:
                        cdt_pre_cont_list, cdt_post_cont_list, dual_pre_cont_list, dual_post_cont_list = [], [], [], []
                        cdt_pre_aff_list, cdt_post_aff_list, dual_pre_aff_list, dual_post_aff_list = [], [], [], []
                        
                        cdt_pre_names, cdt_post_names, dual_pre_names, dual_post_names = [], [], [], []
                        
                        
                        for the_id in study_ids: # use full_ids for 'completed' pts, study_ids for all
                            id_df = sub_df[sub_df['study_id']==the_id]
                            
                            cdt_rows = id_df[id_df['treatment_type']=='cdt_alone']
                            dual_rows = id_df[id_df['treatment_type']=='cdt_and_lt']
                            
                            try:
                                cdt_pre_aff = cdt_rows[cdt_rows['treatment_status']=='pre'][f'aff_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                cdt_pre_aff = np.nan
                            try:
                                cdt_pre_cont = cdt_rows[cdt_rows['treatment_status']=='pre'][f'cont_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                cdt_pre_cont = np.nan
                            
                            try:
                                cdt_post_aff = cdt_rows[cdt_rows['treatment_status']=='post'][f'aff_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                cdt_post_aff = np.nan
                            try:
                                cdt_post_cont = cdt_rows[cdt_rows['treatment_status']=='post'][f'cont_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                cdt_post_cont = np.nan
                            
                            try:
                                dual_pre_aff = dual_rows[dual_rows['treatment_status']=='pre'][f'aff_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                dual_pre_aff = np.nan
                            try:
                                dual_pre_cont = dual_rows[dual_rows['treatment_status']=='pre'][f'cont_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                dual_pre_cont = np.nan
                            
                            try:
                                dual_post_aff = dual_rows[dual_rows['treatment_status']=='post'][f'aff_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                dual_post_aff = np.nan
                            try:
                                dual_post_cont = dual_rows[dual_rows['treatment_status']=='post'][f'cont_{tis}_mean_{sig}_{weight_type}weighted'].iloc[-1] #.mean()
                            except IndexError:
                                dual_post_cont = np.nan
                            
                            
                            try:
                                cdt_pre_ids = cdt_rows[cdt_rows['treatment_status']=='pre']['scan_id'].iloc[-1] #.item()
                            except IndexError:
                                cdt_pre_ids = None
                            try:
                                cdt_post_ids = cdt_rows[cdt_rows['treatment_status']=='post']['scan_id'].iloc[-1] #.item()
                            except IndexError:
                                cdt_post_ids = None
                            try:
                                dual_pre_ids = dual_rows[dual_rows['treatment_status']=='pre']['scan_id'].iloc[-1] #.item()
                            except IndexError:
                                dual_pre_ids = None
                            try:
                                dual_post_ids = dual_rows[dual_rows['treatment_status']=='post']['scan_id'].iloc[-1] #.item()
                            except IndexError:
                                dual_post_ids = None
                                
                            
                            
                            cdt_pre_cont_list.append(cdt_pre_cont)
                            cdt_pre_aff_list.append(cdt_pre_aff)
                            
                            cdt_post_cont_list.append(cdt_post_cont)
                            cdt_post_aff_list.append(cdt_post_aff)
                            
                            dual_pre_cont_list.append(dual_pre_cont)
                            dual_pre_aff_list.append(dual_pre_aff)
                            
                            dual_post_cont_list.append(dual_post_cont)
                            dual_post_aff_list.append(dual_post_aff)  
                            
                            cdt_pre_names.append(cdt_pre_ids)
                            cdt_post_names.append(cdt_post_ids)                        
                            dual_pre_names.append(dual_pre_ids)
                            dual_post_names.append(dual_post_ids)
                            
                        cdt_pre_cont_list = np.array(cdt_pre_cont_list)
                        cdt_post_cont_list = np.array(cdt_post_cont_list)
                        cdt_pre_aff_list = np.array(cdt_pre_aff_list)
                        cdt_post_aff_list = np.array(cdt_post_aff_list)
                        
                        dual_pre_cont_list = np.array(dual_pre_cont_list)
                        dual_post_cont_list = np.array(dual_post_cont_list)
                        dual_pre_aff_list = np.array(dual_pre_aff_list)
                        dual_post_aff_list = np.array(dual_post_aff_list)
                        
                        
                        cdt_aff_change_abs = cdt_pre_aff_list - cdt_post_aff_list
                        cdt_cont_change_abs = cdt_pre_cont_list - cdt_post_cont_list
                        
                        dual_aff_change_abs = dual_pre_aff_list - dual_post_aff_list
                        dual_cont_change_abs = dual_pre_cont_list - dual_post_cont_list
                        
                        #cdt_aff_change_rel = cdt_aff_change_abs / cdt_pre_aff_list
                        #cdt_cont_change_rel = cdt_cont_change_abs / cdt_pre_cont_list
                        
                        #dual_aff_change_rel = dual_aff_change_abs / dual_pre_aff_list
                        #dual_cont_change_rel = dual_cont_change_abs / dual_pre_cont_list
                        
                        response_df[f'cdt_pre_names'] = cdt_pre_names
                        response_df[f'cdt_post_names'] = cdt_post_names
                        response_df[f'dual_pre_names'] = dual_pre_names
                        response_df[f'dual_post_names'] = dual_post_names
                        
                        
                        response_df[f'cdt_pre_cont_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_pre_cont_list
                        response_df[f'cdt_post_cont_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_post_cont_list
                        response_df[f'cdt_delta_cont_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_cont_change_abs
                        #response_df[f'cdt_relative_delta_cont_signal_{tis}_{sig}'] = cdt_cont_change_rel
                        response_df[f'cdt_pre_aff_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_pre_aff_list
                        response_df[f'cdt_post_aff_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_post_aff_list
                        response_df[f'cdt_delta_aff_signal_{tis}_{sig}_{weight_type}weighted'] = cdt_aff_change_abs
                        #response_df[f'cdt_relative_delta_aff_signal_{tis}_{sig}'] = cdt_aff_change_rel
                        
                        response_df[f'dual_pre_cont_signal_{tis}_{sig}_{weight_type}weighted'] = dual_pre_cont_list
                        response_df[f'dual_post_cont_signal_{tis}_{sig}_{weight_type}weighted'] = dual_post_cont_list
                        response_df[f'dual_delta_cont_signal_{tis}_{sig}_{weight_type}weighted'] = dual_cont_change_abs
                        #response_df[f'dual_relative_delta_cont_signal_{tis}_{sig}'] = dual_cont_change_rel
                        response_df[f'dual_pre_aff_signal_{tis}_{sig}_{weight_type}weighted'] = dual_pre_aff_list
                        response_df[f'dual_post_aff_signal_{tis}_{sig}_{weight_type}weighted'] = dual_post_aff_list
                        response_df[f'dual_delta_aff_signal_{tis}_{sig}_{weight_type}weighted'] = dual_aff_change_abs
                        #response_df[f'dual_relative_delta_aff_signal_{tis}_{sig}'] = dual_aff_change_rel
                        
                        
                        labels = ['CDT alone', 'CDT + LT']
                        affs = [cdt_aff_change_abs, dual_aff_change_abs]
                        conts = [cdt_cont_change_abs, dual_cont_change_abs]
                        
                        scan_names = [list(zip(cdt_pre_names, cdt_post_names)), list(zip(dual_pre_names, dual_post_names))]
                        
                        x = np.arange(len(labels))  # the label locations
                        width = 0.2  # the width of the bars
                        
                        fig, ax = plt.subplots(figsize=(8,8))
                        rects1 = ax.bar(x - width/2, [np.nanmean(i) for i in affs], width, label=f'Affected arm', color=(0,0,0,0), edgecolor='mediumseagreen', yerr=[np.nanstd(i) for i in affs])
                        rects2 = ax.bar(x + width/2, [np.nanmean(i) for i in conts], width, label=f'Contralateral arm', color=(0,0,0,0), edgecolor='dodgerblue', yerr=[np.nanstd(i) for i in conts])
                        
                        data_blob = [affs, conts]
                        mover = [-width/2, width/2]
                        colors = ['mediumseagreen', 'dodgerblue']
                        for the_data, move, co in zip(data_blob, mover, colors):
                            #print(f'Outer loop: {co}')
                            # distribute scatter randomly across whole width of bar
                            for i, name_group in zip(range(len(x)), scan_names):
                                #print(f'\tInner loop: len is {len(x)}, {len(scan_names)}')
                                data = the_data[i]
                                exes = x[i] + np.random.random(data.size) * width*0.7 - width / 2 + move
                                ax.scatter(exes, data, color=co, alpha=0.5)
                                
                                here_mean = np.nanmean(data)
                                here_std = np.nanstd(data)
                                here_n = len([i for i in data if not np.isnan(i)])
                                #print(f'\t\t{[len(i) for i in [exes,data,name_group,study_ids]]}')
                                for ex,why,namer,sid in zip(exes,data,name_group,study_ids):
                                    #print(co)
                                    if np.isnan(why):
                                        continue
                                    elif why > here_mean+here_std*1 or why < here_mean-here_std*1:
                                        ax.annotate(f"{sid}:\n{namer[0]} ->\n{namer[1]}", xy=(ex,why), xytext=(20,20),textcoords="offset points", alpha=0.5, size=4,
                                            bbox=dict(boxstyle="round", fc="w", alpha=0.5),
                                            arrowprops=dict(arrowstyle="->"))
                                    
                                
                                #plt.annotate(f'Mean: {round(here_mean,2)}\nn: {here_n}', (x[i]+move, here_mean))
                        
                        # Add some text for labels, title and custom x-axis tick labels, etc.
                        ax.set_ylabel(f'Relative change in {sig.upper()} signal ({weight_type} weighted)')
                        ax.set_title(f'Paired response in {sig.upper()} signal in {tis} to treatment ({weight_type} weighted)')
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels)
                        ax.legend()
                        
                        ax.set_ylim(-0.75,.25)
                        
                        ax.plot(ax.get_xlim(),[0,0],c='red')
                        
                        #ax.bar_label(rects1, padding=3)
                        #ax.bar_label(rects2, padding=3)
                        
                        fig.tight_layout()
                        
                        treatment_response_name = os.path.join(target_aim_folder, f'paired_treatment_response_{sig}_{tis}_{weight_type}weighted.png')
                        plt.savefig(treatment_response_name, dpi=300)
                        
                        plt.close()
                        
                        # save a csv of this info
                        
                response_csv = os.path.join(target_aim_folder, f'response.csv')
                response_df.to_csv(response_csv)
                
                # generate response plots
                def format_axes(fig):
                    for i, ax in enumerate(fig.axes):
                        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
                        ax.tick_params(labelbottom=False, labelleft=False)
                for weight_type in weight_types:        
                    for i,row in response_df.iterrows():
                        sid = row['study_id']
                        
                        plt.rc('xtick', labelsize=5)  
                        # gridspec inside gridspec
                        fig = plt.figure(figsize=(20,10))
                        
                        gs0 = gridspec.GridSpec(1, 2, figure=fig)
                        gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=.4)
                        gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=.4)
                        
                        #gs00 is the contralateral arm
                        #gs01 is the affected arm
                        # within gsx, index 0 is CDT, index 1 is CDT + LT
                        # within CDT or CDT + LT, index 0 is before treatment, index 1 is after
                        
                        for g_plot, arm_type, p_col in zip([gs00, gs01], ['cont', 'aff'], ['mediumseagreen', 'steelblue']):
                            
                            for j, treat_type in enumerate(['cdt', 'dual']):
                                subg = gridspec.GridSpecFromSubplotSpec(6, 2, subplot_spec=g_plot[j], hspace=.6)
                                
                                
            
                                zspec_muscle_axis = fig.add_subplot(subg[0,:])
                                zspec_fat_axis = fig.add_subplot(subg[1,:])
                                
                                zspec_muscle_axis.set_ylim(0,1.2)
                                zspec_fat_axis.set_ylim(0,1.2)
                                
                                zspec_muscle_axis.set_xlim(-4,4)
                                zspec_fat_axis.set_xlim(-4,4)
                                
                                #tiks = np.arange(-4, 4.5, step=0.5)
                                #tiks = [round(i,1) for i in tiks]
                                
                                tiks = np.arange(-5.5, 6, step=0.5)
                                tiks = [round(i,1) for i in tiks]
                                
                                zspec_muscle_axis.set_xticks(tiks)
                                zspec_fat_axis.set_xticks(tiks)
                                
                                zspec_muscle_axis.set_title(f'{arm_type}, {treat_type} (muscle, {weight_type} weighted)')
                                zspec_fat_axis.set_title(f'{arm_type}, {treat_type} (fat, {weight_type} weighted)')                        
                                '''
                                zspec_muscle_axis_uncor = fig.add_subplot(subg[2,:])
                                zspec_fat_axis_uncor = fig.add_subplot(subg[3,:])
                                
                                zspec_muscle_axis_uncor.set_xlim(-5.5,5.5)
                                zspec_fat_axis_uncor.set_xlim(-5.5,5.5)
                                
                                zspec_muscle_axis_uncor.set_xticks(tiks)
                                zspec_fat_axis_uncor.set_xticks(tiks)
                                
                                zspec_muscle_axis_uncor.set_title(f'{arm_type}, {treat_type} (muscle, uncorrected)')
                                zspec_fat_axis_uncor.set_title(f'{arm_type}, {treat_type} (fat, uncorrected)')
                                
                                '''
                                
                                
                                zspec_muscle_axis_uncor_twin = zspec_muscle_axis.twinx()
                                zspec_fat_axis_uncor_twin = zspec_fat_axis.twinx()
                                
                                
                                
                                apt_axes = []
                                noe_axes = []
                                qc_axes = []
                                
                                
                                #seemap = 'jet'
                                
                                seemap = cm.get_cmap('jet', 256)
                                halfmap = cm.get_cmap('jet', 128)
                                newcolors = seemap(np.linspace(0, 1, 256))
                                halfmapcolors = halfmap(np.linspace(0,1,128))
                                gray = np.array([210/256, 210/256, 210/256, 1])
                                badcolor = np.array([1, 1, 1, 1])
                                newcolors[:26, :] = badcolor
                                newcolors[26:128, :] = gray
                                newcolors[128:, :] = halfmapcolors
                                
                                newcmp = ListedColormap(newcolors)
                                
                                
                                cbar_axis = fig.add_subplot(subg[2,:])
                                cbar_axis.set_title(f'APT/NOE colorbar')
                                cbar_axis.set_aspect(0.075)
                                plt.colorbar(cm.ScalarMappable(norm=None, cmap=newcmp), cax=cbar_axis, orientation='horizontal')
                                
                                for (i, state), ls in zip(enumerate(['pre', 'post']), ['solid','dashed']):
                                    
                                    scan_id = row[f'{treat_type}_{state}_names']
                                    
                                    if scan_id is None:
                                        continue
                                    
                                    big_cest_row = cest_df[cest_df['scan_id']==scan_id].iloc[0]
                                    shifts = big_cest_row['chemical_shifts_ppm']
                                    muscle_sigs = big_cest_row[f'muscle_{arm_type}_signals_meannorm_{weight_type}weighted']
                                    fat_sigs = big_cest_row[f'fat_{arm_type}_signals_meannorm_{weight_type}weighted']
                                    
                                    
                                    
                                    shifts_uncor = big_cest_row['chemical_shifts_uncor_ppm']
                                    muscle_sigs_uncor = big_cest_row[f'muscle_{arm_type}_signals_mean_uncor_{weight_type}weighted']
                                    fat_sigs_uncor = big_cest_row[f'fat_{arm_type}_signals_mean_uncor_{weight_type}weighted']
                                    try:
                                        shifts_uncor = [float(i) for i in shifts_uncor[1:-1].split(', ')]
                                    except TypeError:
                                        pass
                                    
                                    try:
                                        muscle_sigs_uncor = [float(i) for i in muscle_sigs_uncor[1:-1].split(', ')]
                                        zspec_muscle_axis_uncor_twin.plot(shifts_uncor, muscle_sigs_uncor, ls=ls, label=f'{state} (uncorrected)', color='red', alpha=0.2)
                                    except TypeError:
                                        pass
                                    
                                    try:
                                        fat_sigs_uncor = [float(i) for i in fat_sigs_uncor[1:-1].split(', ')]
                                        zspec_fat_axis_uncor_twin.plot(shifts_uncor, fat_sigs_uncor, ls=ls, label=f'{state} (uncorrected)', color='red', alpha=0.2)
                                    except:
                                        pass
                                    
                                    
                                    
                                    try:
                                        shifts = [float(i) for i in shifts[1:-1].split(', ')]
                                    except TypeError:
                                        pass
                                    
                                    try:
                                        muscle_sigs = [float(i) for i in muscle_sigs[1:-1].split(', ')]
                                        zspec_muscle_axis.plot(shifts, muscle_sigs, ls=ls, label=state, color=p_col)
                                    except TypeError:
                                        pass
                                    
                                    try:
                                        fat_sigs = [float(i) for i in fat_sigs[1:-1].split(', ')]
                                        zspec_fat_axis.plot(shifts, fat_sigs, ls=ls, label=state, color=p_col)
                                    except:
                                        pass
                                    
                                    zspec_muscle_axis.legend()
                                    zspec_fat_axis.legend()
                                    
                                    #zspec_muscle_axis_uncor_twin.legend()
                                    #zspec_fat_axis_uncor_twin.legend()
                                    
                                    zspec_muscle_axis.set_xlabel('Chemical shift (ppm)')
                                    zspec_muscle_axis.set_ylabel('Normalized signal')
                                    
                                    zspec_fat_axis.set_xlabel('Chemical shift (ppm)')
                                    zspec_fat_axis.set_ylabel('Normalized signal')
                                    
                                    '''
                                    zspec_muscle_axis_uncor.legend()
                                    zspec_fat_axis_uncor.legend()
                                    
                                    zspec_muscle_axis_uncor.set_xlabel('Chemical shift (ppm)')
                                    zspec_muscle_axis_uncor.set_ylabel('Raw signal')
                                    
                                    zspec_fat_axis_uncor.set_xlabel('Chemical shift (ppm)')
                                    zspec_fat_axis_uncor.set_ylabel('Raw signal')
                                    '''
                                    
                                    '''
                                    apt_left_ex = -3.7
                                    apt_right_ex = -3.4
                                    
                                    noe_left_ex = 3.4
                                    noe_right_ex = 3.7
                                    
                                    line_whys = [0, 1]
                                    
                                    for ex in [apt_left_ex, apt_right_ex, noe_left_ex, noe_right_ex]:
                                        zspec_muscle_axis.plot([ex, ex], line_whys, color='black')
                                        zspec_fat_axis.plot([ex, ex], line_whys, color='black')
                                    '''
                                    
                                    fol = os.path.join(master_folder, 'aim3', 'processed', f'{scan_id}_cestdixon_{arm_type}')
                                                    
                                    water_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}_C1_.nii.gz')
                                    fat_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}_C4_.nii.gz')
                                    apt_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}OCEST_APT_{weight_type}.nii.gz')
                                    noe_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}OCEST_OPPAPT_{weight_type}.nii.gz')
                                    s0_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}OCEST_S0_{weight_type}.nii.gz')
                                    fatmask_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}_fatmask.nii.gz')
                                    watermask_image = os.path.join(fol, f'{scan_id}_cestdixon_{arm_type}_musclemask.nii.gz')
                                    
                                    try:
                                        s0_image_loaded = nib.load(s0_image)
                                        s0_image_data = s0_image_loaded.get_fdata()
                                        s0_image_data = np.where(np.isclose(s0_image_data,0), np.nan, s0_image_data)
                                    except FileNotFoundError:
                                        pass
                                    
                                    try:
                                        apt_axis = fig.add_subplot(subg[3,i])
                                        apt_axis.set_title(f'APT ({state})')
                                        apt_axis.set_axis_off()
                                        apt_axis.set_aspect('equal')
                                        apt_image_loaded = nib.load(apt_image)
                                        apt_image_data = apt_image_loaded.get_fdata() / s0_image_data
                                        apt_slice = apt_image_data[:,:,5]
                                        apt_axis.imshow(apt_slice, vmin=0, vmax=1, cmap=newcmp)
                                    except FileNotFoundError:
                                        pass
                                    
                                    try:              
                                        noe_axis = fig.add_subplot(subg[4,i])
                                        noe_axis.set_title(f'NOE ({state})')
                                        noe_axis.set_axis_off()                            
                                        noe_axis.set_aspect('equal')
                                        noe_image_loaded = nib.load(noe_image)
                                        noe_image_data = noe_image_loaded.get_fdata() / s0_image_data
                                        noe_slice = noe_image_data[:,:,5]
                                        noe_axis.imshow(noe_slice, vmin=0, vmax=1, cmap=newcmp)
                                    except FileNotFoundError:
                                        pass
            
                                    try:
                                        qc_axis = fig.add_subplot(subg[5,i])
                                        qc_axis.set_title(f'QC ({state})\n{scan_id}')
                                        qc_axis.set_axis_off()
                                        qc_axis.set_aspect('equal')
                                        water_image_loaded = nib.load(water_image)
                                        water_image_data = water_image_loaded.get_fdata()
                                        water_image_data = np.mean(water_image_data, 3) # time average
                                        water_slice = water_image_data[:,:,5]
                                        qc_axis.imshow(water_slice, cmap='gray')
                                        
                                        fatmask_loaded = nib.load(fatmask_image)
                                        fatmask_data = fatmask_loaded.get_fdata()
                                        fatmask_slice = fatmask_data[:,:,5]
                                        
                                        m_fat = np.ma.masked_where(fatmask_slice != 1, fatmask_slice)
                                        qc_axis.imshow(m_fat, cmap='Greens', interpolation='nearest', alpha=0.5, vmin=0, vmax=2)
                                        
                                        water_image_loaded = nib.load(watermask_image)
                                        water_image_data = water_image_loaded.get_fdata()
                                        
                                        watermask_loaded = nib.load(watermask_image)
                                        watermask_data = watermask_loaded.get_fdata()
                                        watermask_slice = watermask_data[:,:,5]
                                        
                                        m_water = np.ma.masked_where(watermask_slice != 1, watermask_slice)
                                        qc_axis.imshow(m_water, cmap='Reds', interpolation='nearest', alpha=0.5, vmin=0, vmax=2)
                                    except FileNotFoundError:
                                        pass
                                    
                                    
                        
                        
                        plt.tight_layout()
                        
                        
                        tracking_figure = os.path.join(target_aim_folder, f'tracking_{sid}_{weight_type}weighted.png')
                        plt.savefig(tracking_figure, dpi=300)
                        
                        plt.close()
                        #plt.show()
                        #sys.exit()
                        
            
                    
                    '''
                    ## pre/post bar plot
                    cdt_group = sub_df[sub_df['treatment_type'] == 'cdt_alone']
                    dual_group = sub_df[sub_df['treatment_type'] == 'cdt_and_lt']
                    
                    cdt_pre = cdt_group[cdt_group['treatment_status'] == 'pre']
                    cdt_post = cdt_group[cdt_group['treatment_status'] == 'post']
                    
                    dual_pre = dual_group[dual_group['treatment_status'] == 'pre']
                    dual_post = dual_group[dual_group['treatment_status'] == 'post']
            
                    for tis in tissue_types:
                        for sig in sig_types:
                            
                            cdt_pre_sigs_aff = np.array(cdt_pre[f'aff_{tis}_mean_{sig}'])
                            n_cdt_pre_sigs_aff = np.count_nonzero(~np.isnan(cdt_pre_sigs_aff))
                            cdt_pre_sigs_aff_mean = np.nanmean(cdt_pre_sigs_aff)
                            cdt_pre_sigs_aff_std = np.nanstd(cdt_pre_sigs_aff)
                            
                            cdt_pre_sigs_cont = np.array(cdt_pre[f'cont_{tis}_mean_{sig}'])
                            n_cdt_pre_sigs_cont = np.count_nonzero(~np.isnan(cdt_pre_sigs_cont))
                            cdt_pre_sigs_cont_mean = np.nanmean(cdt_pre_sigs_cont)
                            cdt_pre_sigs_cont_std = np.nanstd(cdt_pre_sigs_cont)
                            
            
                            dual_pre_sigs_aff = np.array(dual_pre[f'aff_{tis}_mean_{sig}'])
                            n_dual_pre_sigs_aff = np.count_nonzero(~np.isnan(dual_pre_sigs_aff))
                            dual_pre_sigs_aff_mean = np.nanmean(dual_pre_sigs_aff)
                            dual_pre_sigs_aff_std = np.nanstd(dual_pre_sigs_aff)
                            
                            dual_pre_sigs_cont = np.array(dual_pre[f'cont_{tis}_mean_{sig}'])
                            n_dual_pre_sigs_cont = np.count_nonzero(~np.isnan(dual_pre_sigs_cont))
                            dual_pre_sigs_cont_mean = np.nanmean(dual_pre_sigs_cont)
                            dual_pre_sigs_cont_std = np.nanstd(dual_pre_sigs_cont)
                            
                            
                            
                            
                            
                            cdt_post_sigs_aff = np.array(cdt_post[f'aff_{tis}_mean_{sig}'])
                            n_cdt_post_sigs_aff = np.count_nonzero(~np.isnan(cdt_post_sigs_aff))
                            cdt_post_sigs_aff_mean = np.nanmean(cdt_post_sigs_aff)
                            cdt_post_sigs_aff_std = np.nanstd(cdt_post_sigs_aff)
                            
                            cdt_post_sigs_cont = np.array(cdt_post[f'cont_{tis}_mean_{sig}'])
                            n_cdt_post_sigs_cont = np.count_nonzero(~np.isnan(cdt_post_sigs_cont))
                            cdt_post_sigs_cont_mean = np.nanmean(cdt_post_sigs_cont)
                            cdt_post_sigs_cont_std = np.nanstd(cdt_post_sigs_cont)
                            
            
                            dual_post_sigs_aff = np.array(dual_post[f'aff_{tis}_mean_{sig}'])
                            n_dual_post_sigs_aff = np.count_nonzero(~np.isnan(dual_post_sigs_aff))
                            dual_post_sigs_aff_mean = np.nanmean(dual_post_sigs_aff)
                            dual_post_sigs_aff_std = np.nanstd(dual_post_sigs_aff)
                            
                            dual_post_sigs_cont = np.array(dual_post[f'cont_{tis}_mean_{sig}'])
                            n_dual_post_sigs_cont = np.count_nonzero(~np.isnan(dual_post_sigs_cont))
                            dual_post_sigs_cont_mean = np.nanmean(dual_post_sigs_cont)
                            dual_post_sigs_cont_std = np.nanstd(dual_post_sigs_cont)
                            
                            
                            
                            
                            
                            
                    
                            labels = ['Pre-intervention', 'Post-intervention']
                            cdt_aff_means = [cdt_pre_sigs_aff_mean, cdt_post_sigs_aff_mean]
                            cdt_cont_means = [cdt_pre_sigs_cont_mean, cdt_post_sigs_cont_mean]
                            
                            dual_aff_means = [dual_pre_sigs_aff_mean, dual_post_sigs_aff_mean]
                            dual_cont_means = [dual_pre_sigs_cont_mean, dual_post_sigs_cont_mean]
                            
                            cdt_aff_stds = [cdt_pre_sigs_aff_std, cdt_post_sigs_aff_std]
                            cdt_cont_stds = [cdt_pre_sigs_cont_std, cdt_post_sigs_cont_std]
                            
                            dual_aff_stds = [dual_pre_sigs_aff_std, dual_post_sigs_aff_std]
                            dual_cont_stds = [dual_pre_sigs_cont_std, dual_post_sigs_cont_std]
                            
                            x = np.arange(len(labels))  # the label locations
                            width = 0.17  # the width of the bars
                            
                            fig, ax = plt.subplots(figsize=(8,8))
                            rects1 = ax.bar(x - 3*width/2 - width*0.05, cdt_aff_means, width, label=f'CDT alone, affected (n={[n_cdt_pre_sigs_aff, n_cdt_post_sigs_aff]})', color='royalblue', yerr=cdt_aff_stds)
                            rects2 = ax.bar(x - width/2 - width*0.05, cdt_cont_means, width, label=f'CDT alone, contralateral (n={[n_cdt_pre_sigs_cont, n_cdt_post_sigs_cont]})', color='cornflowerblue', yerr=cdt_cont_stds)
                            rects3 = ax.bar(x + width/2 + width*0.05, dual_aff_means, width, label=f'CDT+LT, affected (n={[n_dual_pre_sigs_aff, n_dual_post_sigs_aff]})', color='indianred', yerr=dual_aff_stds)
                            rects4 = ax.bar(x + 3*width/2 + width*0.05, dual_cont_means, width, label=f'CDT+LT, contralateral (n={[n_dual_pre_sigs_cont, n_dual_post_sigs_cont]})', color='salmon', yerr=cdt_cont_stds)
                            
                            # Add some text for labels, title and custom x-axis tick labels, etc.
                            ax.set_ylabel(f'Mean {sig.upper()} signal')
                            ax.set_title(f'Response in {sig.upper()} signal in {tis} to treatment')
                            ax.set_xticks(x)
                            ax.set_xticklabels(labels)
                            #ax.set_ylim(0.95,1.06)
                            ax.legend()
                            
                            #ax.bar_label(rects1, padding=3)
                            #ax.bar_label(rects2, padding=3)
                            
                            fig.tight_layout()
                            
                            treatment_response_name = os.path.join(target_aim_folder, f'treatment_response_{sig}_{tis}.png')
                            plt.savefig(treatment_response_name, dpi=300)
                            
                            plt.close()
            
                    
                    '''
                    
                
                    """
                    #time courses
                    for the_id in study_ids:
                        target_fig_folder = os.path.join(target_aim_folder, the_id)
                        if os.path.exists(target_fig_folder):
                            shutil.rmtree(target_fig_folder)
                        os.mkdir(target_fig_folder)
                        
                        sub_sub_df = sub_df[sub_df['study_id'] == the_id]
                        
                        for sig_name in sig_names:
                            for roi_n in roi_names:
                                if (roi_n == 'whole' and sig_name != 'fat_frac') or (roi_n != 'whole' and sig_name == 'fat_frac'):
                                    continue
                                prepost = []
                                treatment = []
                                scan_ids = []
                                
                                means_aff = []
                                upper_aff = []
                                lower_aff = []
                                
                                means_cont = []
                                upper_cont = []
                                lower_cont = []
                                
                                for idx, row in sub_sub_df.iterrows():
                                    
                                    if row['treatment_status'] == 'pre':
                                        preposter = 0
                                    elif row['treatment_status'] == 'post':
                                        proposter = 1
                                    else:
                                        raise Exception
                                        
                                    prepost.append(preposter)     
                                    
                                    treater = row['treatment_type']
                                    
                                    treater = treater.upper()
                                    treater = treater.replace('_', ' ')
                                    treater = treater.replace('AND', 'and')
                                        
                                    treatment.append(treater)
                                    
                                    
                                    if sig_name != 'fat_frac':
                                                
                                        means_aff.append(row[f'aff_{roi_n}_mean_{sig_name}'])
                                        upper_aff.append(row[f'aff_{roi_n}_mean_{sig_name}'] + row[f'aff_{roi_n}_std_{sig_name}'])
                                        lower_aff.append(row[f'aff_{roi_n}_mean_{sig_name}'] - row[f'aff_{roi_n}_std_{sig_name}'])
                                        
                                        means_cont.append(row[f'cont_{roi_n}_mean_{sig_name}'])
                                        upper_cont.append(row[f'cont_{roi_n}_mean_{sig_name}'] + row[f'cont_{roi_n}_std_{sig_name}'])
                                        lower_cont.append(row[f'cont_{roi_n}_mean_{sig_name}'] - row[f'cont_{roi_n}_std_{sig_name}'])
                                    else:
                                        means_aff.append(row[f'aff_fat_frac'])
                                        means_cont.append(row[f'cont_fat_frac'])
                                    
                                    scan_ids.append(row['scan_id'])
                        """
                    
                
            
    