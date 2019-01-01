import numpy as np 
import nibabel as nib
import argparse
import sys 
import pdb
import os
import time



from keras import backend as K
from keras.models import *
from tensorflow.python.client import device_lib
from nilearn.image import resample_img



def resize_img(img,target_shape,mask=False, pad=False):
    ''' Resample image to specified target shape '''
    # Define interpolation method
    interp = 'nearest' if mask else 'continuous'
    if not pad:
        # Define resolution
        img_shape = np.array(img.shape[:3])
        target_shape = np.array(target_shape)
        res = img_shape/target_shape
        # Define target affine matrix
        new_affine = np.zeros((4,4))
        new_affine[:3,:3] = np.diag(res)
        new_affine[:3,3] = target_shape*res/2.*-1

        new_affine[3,3] = 1.
        
        # Resample image w. defined parameters
        reshaped_img = resample_img(img,
                                    target_affine=new_affine,
                                    target_shape=target_shape,
                                    interpolation=interp)
    else: # padded/cropped image
        reshaped_img = resample_img(img, 
                                    target_affine=img.affine,
                                    target_shape=target_shape,
                                    interpolation=interp)
    return reshaped_img

''' Dice Coefficient Metric '''
def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def resample_image(img_file, target_shape):

    img = nib.load(img_file) 

    img = resize_img(img, target_shape=(256,320,256), mask=False, pad=True) #resample. 

    img = img.get_data() 

    return img 


def pre_process_image(img_file):

    img_data = resample_image(img_file, (256,320,256)) 

    img_data = np.squeeze(img_data.astype(np.float32))

    img_data = np.expand_dims(img_data, axis=0) 

    img_data = np.expand_dims(img_data, axis=0) 

    min_val = np.min(img_data) 
    max_val = np.max(img_data)

    norm_img_data = (img_data - min_val) / (max_val - min_val + 1e-7) 
    return norm_img_data


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process input images')
    parser.add_argument("input_file")


    args = parser.parse_args()

#    pdb.set_trace()
    if not args.input_file:
        print('Please specify the path of a MRI image for defacing.')
        sys.exit()




    MRI_image = args.input_file


    print('Preproessing input MRI image...')

    MRI_image_data = pre_process_image(MRI_image)
    MRI_image_shape = MRI_image_data.shape

    deepdeface = load_model('model.hdf5', custom_objects={'dice_coefficient': dice_coefficient})
    pdb.set_trace()

    print('Masking %s ....' % (MRI_image))



    print('Starting prediction ...') 
    start_time = time.time() 

    mask_prediction = deepdeface.predict(MRI_image_data) 

    mask_prediction[mask_prediction < 0.5] = 0 
    mask_prediction[mask_prediction >= 0.5] = 1


    mask_prediction = resample_image(mask_prediction, target_shape=MRI_image_shape)


    masked_image = np.multiply(MRI_image_data, mask_prediction)

    print("--- %s seconds ---" % (time.time() - start_time))


    masked_image_save = nib.Nifti1Image(masked_image, nib.load(MRI_image).affine)


    output_file = os.path.splitext(os.path.splitext(os.path.basename(MRI_image))[0])[0] + '_defaced.nii.gz'


    print('Completed! Saving to %s...' % (output_file))

    nib.save(masked_image_save, output_file)

    print('Done.') 












