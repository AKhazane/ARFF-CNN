from __future__ import division
from data_loader import *
import numpy as np
import nibabel as nib
import keras
import pdb
import nilearn 
import random 
from random import randint
import skimage as sk
from skimage import transform 
from skimage import util 



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, examples, labels, batch_size=32, dim=(256,256), n_channels=1
                 , shuffle=True, third_dimension=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = examples
        self.n_channels = n_channels
        self.third_dimension = third_dimension
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_ys_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_ys_temp, self.third_dimension)
        return X, y


    def data_augmentation(self, x, y):
        choice = randint(0, 2) 
        if choice == 0:
#            random_degree = random.uniform(-75, 75)
            x = np.flipud(x)
            y = np.flipud(y)
        elif choice == 1:
            x = np.fliplr(x) 
            y = np.fliplr(y)
        return x,y 
    
    def resize_image(self, image, mask=False):
          #new_dims = tuple((image.shape[0] + (self.dim[0] - image.shape[0]), image.shape[1], image.shape[2]))
        if mask:
            img = nilearn.image.resample_img(target_shape=(256, 320, 256)) 
        #if mask:
        #    new_image = np.ones(new_dims) # somewhat odd because a sliver of white will appear around the mask in the GT image 
        #else:
        #    new_image = np.zeros(new_dims) 
        #new_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image 
        return img


    def resample_image(nifti_img, specified_shape):

        img = sitk.ReadImage(nifti_img)
        img_data = sitk.GetArrayFromImage(img)

        shape = img_data.shape
        dimension = img.GetDimension() 
        target_shape = specified_shape

        reference_physical_size = np.zeros(dimension)
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

        reference_origin = np.zeros(dimension)
        reference_direction = np.identity(dimension).flatten()
        reference_size = specified_shape # Arbitrary sizes, smallest size that yields desired results. 
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size)]

        reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        resampled_img_data = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
        resampled_img_data = np.swapaxes(sitk.GetArrayFromImage(resampled_img_data), 0, -1) 

        return resampled_img_data

    
    def getMaskData(self, normal, defaced):
        normalized_norm = ((normal - np.min(defaced))/
                      (np.max(normal)-np.min(defaced)))
        delta = defaced - normalized_norm
        delta[delta >= 0] = 1.0
        delta[delta < 0] = 0.0
        return delta

    def normalizeImg2(self, x):
        min_val = np.min(x) 
        max_val = np.max(x)

        norm_x = (x - min_val) / (max_val - min_val + 1e-7) 
        return norm_x 

    def normalizeImg(self, x):
        # Normalize x
        mean_val = np.mean(x)
        max_val = np.max(x)
        min_val = np.min(x)
        norm_x = (x-mean_val)/(max_val - min_val + 1e-7)
        return norm_x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_ys_temp, third_dimension=False):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, self.dim[0], self.dim[1], self.dim[2]))
        y = np.empty((self.batch_size, self.n_channels, self.dim[0], self.dim[1], self.dim[2]))
        # Need to split larger image into slices
       
        x_slice_idx = 0
        y_slice_ix = 0
        num_slices = 60
        # Generate data
        # Num Slcies
        for i, ID in enumerate(list_IDs_temp):
            # Load scan data for raw scan & mask
            if third_dimension:
                # set channels as first dimension for 3D Unet.
                #pdb.set_trace() 
                 
                # Augment incoming training data according to self.augmentors 
#                x_data = np.expand_dims(np.squeeze(nib.load(ID).get_data().astype(np.float32)), axis=0)
#                y_data = np.expand_dims(np.squeeze(nib.load(list_ys_temp[i]).get_data().astype(np.float32)), axis=0)
                raw_x, raw_y = np.squeeze(resample_image(ID, (160,160,160))).astype(np.float32), np.squeeze(resample_image(list_ys_temp[i], (160,160,160))).astype(np.float32)

                raw_x, raw_y = self.data_augmentation(raw_x, raw_y)

                x_data = np.expand_dims(raw_x, axis=0)
                y_data = np.expand_dims(raw_y, axis=0)
#                x_data = self.resize_image(x_data) 
#                y_data = self.resize_image(y_data) 
                #if x_data.shape[0] < self.dim[0]:
#                    x_data = self.resize_image(x_data) 
#                    y_data = self.resize_image(y_data, mask=True)

                #x_data = self.normalizeImg(x_data) 
                #y_data = self.getMaskData(x_data, y_data)
                x_data = self.normalizeImg2(x_data)
                
                X[x_slice_idx,] = x_data
                y[y_slice_ix,] = y_data

                x_slice_idx += 1
                y_slice_ix += 1

            else:

                for x_idx in range(x_data.shape[2]):
                    X[xslice_idx,] = x_data[:, :, x_idx, np.newaxis]
                    xslice_idx+=1
                    if xslice_idx == num_slices:
                        break

                
                for y_idx in range(y_data.shape[2]):
                    y[yslice_idx,] = y_data[:, :, y_idx, np.newaxis]
                    yslice_idx+=1
                    if yslice_idx == num_slices:
                        break 
                if xslice_idx == num_slices and yslice_idx == num_slices:
                    break 
        x1 = X[:x_slice_idx, :, :, :].astype(np.float32)
        y1 = y[:y_slice_ix, :, :, :].astype(np.float32)
        return x1, y1
      
