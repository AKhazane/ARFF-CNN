import os 

import unet
import json
from keras import backend as K
from keras.models import *
from keras.utils import multi_gpu_model
from data_loader import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generation import DataGenerator
import nibabel as nib
import pdb 
import tensorflow as tf
from metrics import dice_coefficient
import argparse
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--validation_test', default=False, action='store_true') 

parser.add_argument('--restore', default=False, action='store_true') 

parser.add_argument('--speed_test', default=False, action='store_true') 

parser.add_argument('--test', default=False, action='store_true') 

parser.add_argument('--checkpoint', default='unet_3d_bse.hdf5', action='store') 

parser.add_argument('--convert_to_tensorflow', default=False, action='store_true') 

args = parser.parse_args()

def save_img(img, fn = 'd_mask.nii'):
	img_nii = nib.Nifti1Image(img, np.eye(4))
	img_nii.to_filename(os.path.join('.', fn))
	return True

def save_prediction(file_names, predictions, directory):
	for idx, fn in enumerate(file_names):
  #              pdb.set_trace()
		pred_fn = directory + os.path.basename(os.path.normpath(os.path.splitext(fn)[0]))
		pred_fn = pred_fn.replace('_mask.nii', '_pred_mask.nii.gz')
		save_img(predictions[idx], fn=pred_fn) 
		print(pred_fn, 'saved.')



def speed_test(checkpoint_file='unet_3d_bse.hdf5'):
	print(checkpoint_file)
	print('Setting up speed test...')
	partition = {}
	model = load_model(checkpoint_file, custom_objects={'dice_coefficient': dice_coefficient}) 

	(_,
	    _,
	   	_,
	    _,
	    partition['x_test'],
	    partition['y_test'])  = load_data('test_set_mri', split=(0,0,100), DEBUG=False, third_dimension=True)

	print('Number of images to mask', len(partition['x_test']))

	params = {
			'dim': (160,256,256),
        	'batch_size': 1,
        	'n_channels': 1,
        	'shuffle': False,
            'third_dimension': True
	     	}

	testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
	avg_time = [] 
#	pdb.set_trace()
	print('Starting test.') 
	for index, filename in enumerate(testing_generator):
		x_batch, _ = testing_generator[index]
		if index >= len(testing_generator.list_IDs):
			break
		file = testing_generator.list_IDs[index]
		#print('Predicting', file)
		#norm_image = nib.load(file.replace('test_set_mri/', 'data/')).get_data()
		#norm_image = np.swapaxes(testing_generator.resize_image(np.swapaxes(norm_image, 0, -1)), 0, -1)
		print('Predicting', file)
		start = time.time()
		predicted_img = model.predict_on_batch(x=x_batch)
		end = time.time() 
		avg_time.append(end - start)
		#predicted_mask = np.squeeze(predicted_mask)
		# alternative_mask = predicted_mask.copy() 
		#less_indices = predicted_mask < 0.5
		#higher_indics = predicted_mask >= 0.5
		#predicted_mask[less_indices] = 0
		#predicted_mask[higher_indics] = 1
		#predicted_mask = np.swapaxes(predicted_mask, 0, -1) 
#         print(norm_data.shape, alternative_mask.shape)
		#norm_output = predicted_mask * norm_image
		#end = time.time()
		#avg_time.append((end - start))
		print(end - start)
	print('Average prediction speed(s):', np.mean(avg_time))




def predict(model, validation_generator, test_set):
	for i in range(len(validation_generator)):
		if i == 5:
			break
		x_batch, _ = validation_generator[i] 
		start = time.time() 
		predicted_mask = model.predict_on_batch(x=x_batch)
		end = time.time() 
		print('Prediction time:', end - start) 
		save_prediction([test_set[i]], predicted_mask)


def evaluate(validation=True, checkpoint='unet_3d_bse.hdf5'):
        #pdb.set_trace()
	partition = {}
#        print('Using checkpoint: %s'  % (checkpoint)) 
	print('Using checkpoint %s' % checkpoint) 
	model = load_model('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch.hdf5', custom_objects={'dice_coefficient': dice_coefficient}) 
	#model.load_weights('unet_3d_regression.hdfs')
	#pdb.set_trace()
	(_,
	    _,
	    partition['x_val'],
	    partition['y_val'],
	    _,
	    _)  = load_data('test_set' if not validation else 'validation_set', split=(0,100,0), DEBUG=True, third_dimension=True)

	print('Number of images to mask', len(partition['x_val']))
	params = {
		'dim': (256,320,256),
        	'batch_size': 1,
        	'n_channels': 1,
        	'shuffle': False,
                'third_dimension': True
	     	}

	#pdb.set_trace()
	#pdb.set_trace()
	validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
	for index, filename in enumerate(validation_generator):
		x_batch, _ = validation_generator[index]
		print('dealing with file', validation_generator.list_IDs[index])
		predicted_mask = model.predict_on_batch(x=x_batch)
		if validation:
			save_prediction([partition['y_val'][index]], predicted_mask, 'validation_predictions/')
		else:
			save_prediction([partition['y_val'][index]], predicted_mask, 'test_predictions/')



def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        #pdb.set_trace()
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph	


def convert_to_pb(model_file):
	K.clear_session()
        #pdb.set_trace() 
	model = load_model(model_file, custom_objects={'dice_coefficient': dice_coefficient})
	
	frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, "graphs", "best_3d_model.pb", as_text=False)
	print('Successfully translated keras model to tensorflow graph session!')

def train(restore=False):
 
#	K.clear_session()
	#pdb.set_trace()
	partition = {}
	if not restore:
		model = unet.unet((1, 256, 320, 256))
#                model = multi_gpu_model(model, gpus=[0,1], cpu_merge=True, cpu_relocation=False)

		print('Instantiated new 3D-Unet') 

	if restore:
		pdb.set_trace() 
#         base_model = load_model('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch.hdf5', custom_objects={'dice_coefficient': dice_coefficient})
#         s_config = base_model.get_config()
#         model_idx = np.random.choice(500)
#         name_mapping = {v['name']: '%04d_%04d' % (model_idx, c_idx) for c_idx,v in enumerate(s_config['layers'])}
#         raw_str = json.dumps(s_config)
#         for k,v in name_mapping.items():
#             raw_str = raw_str.replace('"{}"'.format(k), '"{}"'.format(v))
#         n_config = json.loads(raw_str)
# #            for index in range(len(n_config['layers'])):
#         n_config['layers'][0]['config']['batch_input_shape'] = (1, 1, 256, 320, 256)

#         fix_model = Model.from_config(n_config)
#         fix_model.load_weights('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch.hdf5')
#         fix_model.save('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch_ALTERED_2.hdf5')




		old_model = load_model('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch.hdf5', custom_objects={'dice_coefficient': dice_coefficient}) 
        base_model = unet.unet((1, 1, 256, 320, 256)) 
        for new_layer, layer in zip(base_model.layers[1:], old_model.layers[1:]):
            new_layer.set_weights(layer.get_weights())
        s_config = base_model.get_config()
        model_idx = np.random.choice(500)
        name_mapping = {v['name']: '%04d_%04d' % (model_idx, c_idx) for c_idx,v in enumerate(s_config['layers'])}
        raw_str = json.dumps(s_config)
        for k,v in name_mapping.items():
            raw_str = raw_str.replace('"{}"'.format(k), '"{}"'.format(v))
        n_config = json.loads(raw_str)
#            for index in range(len(n_config['layers'])):

        fix_model = Model.from_config(n_config)
        fix_model.load_weights('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch.hdf5')
        fix_model.save('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch_ALTERED_2.hdf5')

         #       pdb.set_trace()
#		new_model.save('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_third_epoch_ALTERED.hdf5')#
		return
		#model.load_weights('unet_3d_binary_cross_entropy.hdfs')
		print('Restored 3D-Unet from latest HDF5 file.')  

	print(model.summary())
	(partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
    partition['y_test'])  = load_data('data', split=(95,5,0), DEBUG=True, third_dimension=True)


	params = {
		    'dim': (256,320,256),
	            'batch_size': 1,
	            'n_channels': 1,
	            'shuffle': True,
                    'third_dimension': True
             }

        #pdb.set_trace()
	training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
	validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
#	testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
	print('Loaded Data')


	model_checkpoint = ModelCheckpoint('unet_3d_bse_ONE_EPOCH_JUST_data_augmentation_fourth_epoch.hdf5', monitor='loss',verbose=1, save_best_only=True)

	model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    #steps_per_epoch = 1,
                    validation_steps = 1,
	  	    epochs=1,
                    #max_queue_size=3,
		    callbacks = [model_checkpoint],
		    use_multiprocessing=True,
		    workers=6,
                    verbose=1)
	model.save_weights('unet_3d_binary_cross_entropy_JUST_data_augmentation.hdf5')

	print('Predicting ...')
	predict(model, validation_generator, partition['y_val'])
#




if __name__ == '__main__':
	args = parser.parse_args()
# 	pdb.set_trace()
        if args.convert_to_tensorflow:
		convert_to_pb(args.checkpoint)
	elif args.validation_test:
		evaluate(validation=True, checkpoint=args.checkpoint)
	elif args.test:
		evaluate(validation=False, checkpoint=args.checkpoint) 
	elif args.restore:
		train(True) 
	elif args.speed_test:
		speed_test('unet_regres_best.hdfs')
	else:
		train() 
