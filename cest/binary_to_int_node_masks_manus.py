#!/usr/bin/env python3

import os
import glob

import nibabel as nib
from skimage import morphology

f1 = '/Users/skyjones/Desktop/cest_processing/data/aim1_for_manus_v2/metastasis/'
f2 = '/Users/skyjones/Desktop/cest_processing/data/aim1_for_manus_v2/no_metastasis/'

fs = [f1, f2]

for f in fs:

    subs = glob.glob(os.path.join(f, '*/'))
    for sub in subs:
        files = glob.glob(os.path.join(sub, '*'))
        mask_list = [i for i in files if 'mask' in i.lower()]
        mask = mask_list[0]
        mask_bname = os.path.basename(os.path.normpath(mask))
        mask_bname_simp = mask_bname.split('.')[0]
    
        node_im_binary = nib.load(mask)
        node_im_binary_data = node_im_binary.get_fdata()
        node_im_int_data = morphology.label(node_im_binary_data)
        
        node_im_int = nib.Nifti1Image(node_im_int_data, node_im_binary.affine, node_im_binary.header)
        
        out_fol = os.path.dirname(os.path.normpath(mask))
        out_path = os.path.join(out_fol, f'{mask_bname_simp}_int.nii.gz')
        
        nib.save(node_im_int, out_path)
        #print(f'{mask} --> {mask_bname_simp}')
        #print(f'\t{out_path}')
                
            

                    
            
    
    