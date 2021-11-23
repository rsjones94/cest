#!/usr/bin/env python3

import os
import glob
import shutil
import datetime
import operator
import sys
import copy

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

import preprocess as pp

mpl.rcParams.update({'errorbar.capsize': 2})

# input
master_folder = '/Users/skyjones/Desktop/cest_processing/data/working/'
cest_spreadsheet = '/Users/skyjones/Desktop/cest_processing/check.csv'

#output
data_out = '/Users/skyjones/Desktop/cest_processing/data/working/results_aims234.csv'
figure_folder = '/Users/skyjones/Desktop/cest_processing/data/working/figures/'

do_prep = True
do_figs = True



##########

def _lorentzian_mono(x, amp1, cen1, wid1, offset1):
        return (amp1*wid1**2/((x-cen1)**2+wid1**2)) + offset1

weight_types = ['water', 'fat']

if do_prep:
    cest_df = pd.read_csv(cest_spreadsheet)
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
            #b0_image = os.path.join(fol, f'{cest_id}_cestdixon_{arm}_C6_.nii.gz') # this is actually just the water-weighted b0
            
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
            
            fat_slice = fat_image_data[:,:,5]
            water_slice = water_image_data[:,:,5]
                                
            fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(8, 12))
            
            for axrow in axs:
                for ax in axrow:
                    ax.axis('off')
            
            # make segmentations
            water_mask, fat_mask, combined_mask, intermediates = pp.segment_muscle_and_fat(water_image_data, fat_image_data)
            
            fat_mask_raw = intermediates['fat_mask_raw']
            water_mask_raw = intermediates['water_mask_raw']
            
            thetas = intermediates['thetas']
            edge_mask = intermediates['edge_mask']
            hole_mask = intermediates['hole_mask']
            flat_intersect = intermediates['flat_intersect']
            raw_skeleton = intermediates['raw_skeleton']
            skeleton = intermediates['skeleton']
            combined_mask_raw = intermediates['combined_mask_raw']

            fat_mask_raw_slice = fat_mask_raw[:,:,5]
            
            axs[0][0].imshow(fat_slice, cmap='gray')
            axs[0][0].set_title('fat-weighted')
            axs[0][1].imshow(fat_mask_raw_slice, cmap='binary')
            
            
            water_mask_raw_slice = water_mask_raw[:,:,5]
            
            axs[1][0].imshow(water_slice, cmap='gray')
            axs[1][0].set_title('water-weighted')
            axs[1][1].imshow(water_mask_raw_slice, cmap='binary')
            
                
            
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

            theta = thetas[:,:,5]
            axs[2][0].imshow(theta, cmap='twilight_shifted', vmin=-180, vmax=180, interpolation='nearest')
            axs[2][0].set_title('Fat-weighted directions')
            
            hillshade_image_name = os.path.join(analysis_folder, id_figure_folder, f'{num_id}_fat_hillshade_{arm}')
    
    
            combined_mask_raw_slice = combined_mask_raw[:,:,5]
            
            axs[2][1].imshow(combined_mask_raw_slice, cmap=cmap, norm=norm, interpolation='nearest')
            axs[2][1].set_title('combined mask')
            

            

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
            
            
            axs[3][1].imshow(raw_skeleton, cmap='Reds', vmin=0, vmax=2, interpolation='nearest')
            axs[3][1].imshow(skeleton, cmap='Greens', vmin=0, vmax=2, interpolation='nearest')
            
            
            combined_mask_slice = combined_mask[:,:,5]
            
    
            my_cmap = copy.copy(plt.cm.get_cmap('inferno'))
            my_cmap.set_bad(alpha=0)
            

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
            
            
            ##### now we're writing extracted data out
            
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
                        sig_copy[sig_copy == 0] = np.nan # remove zero signal wierdery
                        
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
                            fat_mask_slice = fat_mask[:,:,the_index]
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
                        
                    ## finally, fit a lorentzian to the water peak
                    
                                                    
                    # optimiziation initial params for lorentzian
                    if weight_type == 'water':
                        amp_guess = -1
                        center_guess = 0
                        width_guess = 2
                        offset_guess = 1
                    elif weight_type == 'fat':
                        amp_guess = -0.5
                        center_guess = -3
                        width_guess = 2
                        offset_guess = 1
                    
                    try:
                        popt_lor, pcov_lor = scipy.optimize.curve_fit(_lorentzian_mono, shifts, means_norm, [amp_guess, center_guess, width_guess, offset_guess])
                        perr_lor = np.sqrt(np.diag(pcov_lor))
                        
                        opt_sigs = _lorentzian_mono(np.array(shifts), popt_lor[0], popt_lor[1], popt_lor[2], popt_lor[3])
                        for idx in the_rows.index:
                            cest_df.at[idx, f'{roi_n}_{arm}_lorentzian_sigs_{weight_type}weighted'] = str(opt_sigs)
                            cest_df.at[idx, f'{roi_n}_{arm}_lorentzian_amp_{weight_type}weighted'] = str(popt_lor[0])
                            cest_df.at[idx, f'{roi_n}_{arm}_lorentzian_center_{weight_type}weighted'] = str(popt_lor[1])
                            cest_df.at[idx, f'{roi_n}_{arm}_lorentzian_width_{weight_type}weighted'] = str(popt_lor[2])
                            cest_df.at[idx, f'{roi_n}_{arm}_lorentzian_offset_{weight_type}weighted'] = str(popt_lor[3])
                    except ValueError:
                        pass
    
                    
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

                        
            
            # add QoL indicators
            
            #qol_inds = ['ldex']
            qol_inds = ['ldex', 'bmi', 'uefi', 'dash', 'psfs']
            
            temporals = ['pre', 'post']
            ttypes = ['dual', 'cdt']
            
            for i,row in response_df.iterrows():
                for ttype in ttypes:
                    for temporal in temporals:
                        names_col = f'{ttype}_{temporal}_names'
                        name = row[names_col]
                        for qol in qol_inds:
                            response_col = f'{ttype}_{temporal}_{qol}'
                            try:
                                response = float(cest_df[cest_df['scan_id']==name][qol])
                            except TypeError:
                                continue
                            response_df.at[i,response_col] = response
                            
                    for qol in qol_inds:
                        delta_col = f'{ttype}_delta_{qol}'
                        pre_col = f'{ttype}_pre_{qol}'
                        post_col = f'{ttype}_post_{qol}'
                        response_df[delta_col] = response_df[post_col] - response_df[pre_col]
                        
            
            
            #plot QoL changes against signal changes
            
            sigs = ['noe', 'apt']
            arm_types = ['cont', 'aff']
            arm_type_cols = ['green', 'blue']
            tissue_types = ['fat', 'muscle']
            weight_types = ['fat', 'water']
            treat_types = ['dual', 'cdt']

            
            for qol in qol_inds:
                for wtype in weight_types:
                    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))
                    figname = os.path.join(target_aim_folder, f'delta_{qol}_{wtype}weighted.png')
                    for ttype, idex in zip(tissue_types, [0,2]):
                        for ki,sig in enumerate(sigs):
                            for kj,trtype in enumerate(treat_types):
                                ax = axs[ki+idex][kj]
                                ax.set_title(f'{ttype}, {sig}, {trtype}')
                                for arm_type, arm_type_col in zip(arm_types, arm_type_cols):
                                    xcol = f'{trtype}_delta_{arm_type}_signal_{ttype}_{sig}_{wtype}weighted'
                                    ycol = f'{trtype}_delta_{qol}'
                                    
                                    xdat = response_df[xcol]
                                    ydat = response_df[ycol]
                                    
                                    ax.scatter(xdat, ydat, color=arm_type_col, label=arm_type, alpha=0.5, edgecolors='black')
                                    ax.set_xlabel('Normalized signal change (a.u.)')
                                    ax.set_ylabel(f'{qol} change')
                                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(figname)
                                    
                
            # save a csv of this info
                    
            response_csv = os.path.join(target_aim_folder, 'response.csv')
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
                            
                                                                
                                

                                
                            # optimiziation initial params for lorentzian
                            if weight_type == 'water':
                                amp_guess = -1
                                center_guess = 0
                                width_guess = 2
                                offset_guess = 1
                            elif weight_type == 'fat':
                                amp_guess = -0.5
                                center_guess = -3
                                width_guess = 2
                                offset_guess = 1
                            
                            for (i, state), ls, marker in zip(enumerate(['pre', 'post']), ['solid','dashed'], ['o','v']):
                                
                                scan_id = row[f'{treat_type}_{state}_names']
                                
                                if scan_id is None:
                                    continue
                                
                                big_cest_row = cest_df[cest_df['scan_id']==scan_id].iloc[0]
                                shifts = big_cest_row['chemical_shifts_ppm']
                                muscle_sigs = big_cest_row[f'muscle_{arm_type}_signals_meannorm_{weight_type}weighted']
                                fat_sigs = big_cest_row[f'fat_{arm_type}_signals_meannorm_{weight_type}weighted']
                                
                                muscle_sigs_lorentzian = big_cest_row[f'muscle_{arm_type}_lorentzian_sigs_{weight_type}weighted']
                                fat_sigs_lorentzian = big_cest_row[f'fat_{arm_type}_lorentzian_sigs_{weight_type}weighted']
                                
                                muscle_lorentzian_gamma = big_cest_row[f'muscle_{arm_type}_lorentzian_width_{weight_type}weighted']
                                fat_lorentzian_gamma = big_cest_row[f'fat_{arm_type}_lorentzian_width_{weight_type}weighted']
                                
                                
                                
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
                                    muscle_sigs_lorentzian = [float(i) for i in muscle_sigs_lorentzian[1:-1].split(', ')]
                    
                                    zspec_muscle_axis.plot(shifts, muscle_sigs_lorentzian, ls=ls, color='gray', alpha=0.5, lw=1)
                                    
                                    zspec_muscle_axis.plot(shifts, muscle_sigs, ls=ls, label=f'{state} ($\gamma$={round(muscle_lorentzian_gamma,2)})', color=p_col)
                                    #zspec_muscle_axis.scatter(shifts, muscle_sigs, marker=marker, label=None, color=p_col, lw=2, s=2)

                                except TypeError:
                                    pass
                                
                                try:
                                    fat_sigs = [float(i) for i in fat_sigs[1:-1].split(', ')]
                                    fat_sigs_lorentzian = [float(i) for i in fat_sigs_lorentzian[1:-1].split(', ')]
                                    
                                    zspec_fat_axis.plot(shifts, fat_sigs_lorentzian, ls=ls, color='gray', alpha=0.5, lw=1)
                                    
                                    zspec_fat_axis.plot(shifts, fat_sigs, ls=ls, label=f'{state} ($\Gamma$={round(fat_lorentzian_gamma,2)})', color=p_col)
                                    #zspec_fat_axis.scatter(shifts, fat_sigs, marker=marker, label=None, color=p_col, lw=2, s=2)
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
                    
                
            
    