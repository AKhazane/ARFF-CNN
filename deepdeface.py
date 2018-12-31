import numpy as np 
import nibabel as nib
import argparse
import sys 
import pdb
import os
import time



from keras import backend as K
from keras.models import *


''' Dice Coefficient Metric '''
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def pre_process_image(img_file):

    img_data = np.squeeze(nib.load(img_file).get_data().astype(np.float32))

    img_data = np.expand_dims(img_data, axis=0) 

    img_data = np.expand_dims(img_data, axis=0) 

    min_val = np.min(img_data) 
    max_val = np.max(img_data)

    norm_img_data = (img_data - min_val) / (max_val - min_val + 1e-7) 
    return norm_img_data 




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input images')
    parser.add_argument("input_file")


    args = parser.parse_args()

    pdb.set_trace()
    if not args.input_file:
        print('Please specify the path of a MRI image for defacing.')
        sys.exit()

    MRI_image = args.input_file


    print('Preproessing input MRI image...')

    MRI_image_data = pre_process_image(MRI_image)

    deepdeface = load_model('model.hdf5', custom_objects={'dice_coefficient': dice_coefficient})


    print('Masking %s ....' % (MRI_image))



    print('Starting prediction ...') 
    start_time = time.time() 

    mask_prediction = deepdeface.predict(MRI_image_data) 

    mask_prediction[mask_prediction < 0.5] = 0 
    mask_prediction[mask_prediction >= 0.5] = 1


    masked_image = np.multiply(MRI_image_data, mask_prediction)

    print("--- %s seconds ---" % (time.time() - start_time))


    masked_image_save = nib.Nifti1Image(masked_image, nib.load(MRI_image).affine)


    output_file = os.path.splitext(os.path.splitext(os.path.basename(MRI_image))[0])[0] + '_defaced.nii.gz'


    print('Completed! Saving to %s...' % (output_file))

    nib.save(masked_image_save, output_file)

    print('Done.') 












