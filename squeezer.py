from os import listdir
from os.path import isfile, joins
import nib

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
	nifti_img = nib.load(file) 
	nifti_img_affine = nifti_img.affine
	nifti_data = np.squeeze(nifti_img.get_data())

	squeezed_nifit = nib.Nifti1Image(data, nifti_img_affine)

	nib.save(squeezed_nifit, filename)
