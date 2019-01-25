from os import listdir
from os.path import isfile, join
import nibabel as nib
import numpy as np
import os
import pdb
mypath = os.getcwd() + '/validation_set/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
	nifti_img = nib.load(mypath + filename) 
#        if len(nifti_img.get_data().shape) != 3:
#            print(nifti_img.get_data().shape)
#            print(filename)
#            print('-------------')
	nifti_img_affine = nifti_img.affine

	nifti_data = np.squeeze(nifti_img.get_data())

	squeezed_nifit = nib.Nifti1Image(nifti_data, nifti_img_affine)
#
	nib.save(squeezed_nifit, mypath + filename)
