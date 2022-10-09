import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import scipy.ndimage
import matplotlib.pyplot as plt



ct_file = 'your/file/path.nii.gz'
diff_file = 'your/file/path2.nii.gz'
# assuming ct and diff data are in the same imaging space

# load the ct data in
ct_loaded = nib.load(ct_file)
ct_data = ct_loaded.get_fdata()

# load the diffusion data in
diff_loaded = nib.load(diff_file)
diff_data = diff_loaded.get_fdata()
    

# smooth your ct data
ct_smoothed = scipy.ndimage.gaussian_filter(ct_data, sigma=2)

# apply otsu's method to your ct data and make a mask. you may have to fiddle with the threshold
thresh = threshold_otsu(ct_smoothed)
mask = ct_data > thresh

# overlay the mask on your diffusion data: make everything not under the mask a nan
diff_masked = diff_data.copy()
diff_masked[mask!=1] = np.nan

# now you can work with your diffusion data. idk what you want to do.
# let's get the mean of each distinct contiguous masked region
labeled = scipy.ndimage.label(mask)
labels = labeled.unique()

for label in labels:
    region_masked = diff_data.copy()
    region_masked[labeled!=label] = np.nan
    region_mean = np.nanmean(region_masked)
    print(f'Region {label} mean: round(region_mean,4)')
    
    
# also good to make a quality control image
# you may want to play around with axes, aspect ratio etc.
fig, axs = plt.subplots(1,3,figsize=(12,8))
slice_index = 120

orig_ax = axs[0]
orig_slice = ct_data[:,:,slice_index]
orig_ax.imshow(orig_slice)
orig_ax.set_title('Original CT')

smoothed_ax = axs[0]
smoothed_slice = ct_smoothed[:,:,slice_index]
smoothed_ax.imshow(smoothed_slice)
smoothed_ax.set_title('Smoothed CT')

mask_ax = axs[0]
diff_slice = diff_data[:,:,slice_index]
mask_slice = mask[:,:,slice_index]
mask_ax.imshow(diff_slice, cmap='gist_gray')
mask_ax.imshow(mask_slice, alpha=0.5)
mask_ax.set_title('Diffusion with regions')

fig.tight_layout()
fig.show()
