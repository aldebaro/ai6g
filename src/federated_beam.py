import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import scipy.io as sio
import numpy as np
import tensorflow as tf
import scipy.io as sio
import numpy as np
import scipy.stats
import math
from dataloader import LidarDataset2D
from models import Lidar2D


################################Arguments################################
MONTECARLO = 1
NUM_VEHICLES = 2
LOCAL_EPOCHS = 1
AGGREGATION_ROUNDS = 30

BATCH_SIZE = 16
SHUFFLE_BUFFER = 20
PREFETCH_BUFFER=10

lidar_training_path = ["lidar_input_train.npz", "lidar_input_validation.npz"] ###Raymobtime s008
beam_training_path = ["beams_output_train.npz", "beams_output_validation.npz"] ###Raymobtime s008

lidar_test_path = ["lidar_input_test.npz"] ###Raymobtime s009
beam_test_path = ["beams_output_test.npz"] ###Raymobtime s009


################################Functions################################
def get_local_dataset(lidar_path, beam_path, num_vehicles, vehicle_ID):
    training_data = LidarDataset2D(lidar_path, beam_path)
    training_data.lidar_data = np.transpose(training_data.lidar_data, (0, 2, 3, 1))
    x=training_data.lidar_data
    xx = x[vehicle_ID*int(x.shape[0]/num_vehicles):(vehicle_ID+1)*int(x.shape[0]/num_vehicles),:,:,:] ###Split Lidar Data
    y = training_data.beam_output
    yy = y[vehicle_ID*int(y.shape[0]/num_vehicles):(vehicle_ID+1)*int(y.shape[0]/num_vehicles),:] ###Split Beam Labels
    
    dataset_train = tf.data.Dataset.from_tensor_slices((list(xx.astype(np.float32)),list(yy.astype(np.float32))))
    #sio.savemat('label'+str(k)+'.mat',{'label'+str(k):yy})
    return dataset_train

def get_test_dataset(lidar_path, beam_path):
    test_data = LidarDataset2D(lidar_path, beam_path)
    test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))
    dataset_test = tf.data.Dataset.from_tensor_slices((list(test_data.lidar_data.astype(np.float32)),list(test_data.beam_output.astype(np.float32))))
    return dataset_test

def preprocess(dataset):
  def batch_format_fn(element1,element2):
    return collections.OrderedDict(x=element1, y=element2)
  return dataset.repeat(LOCAL_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)    

def create_keras_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(20, 200, 1)),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same'),# kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same'),# kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 2, padding='same'),# kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 1, padding='same'),# kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(5, 3, 2, padding='same'),#, kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Conv2D(1, 3, (1, 2), padding='same'),#, kernel_initializer=initializers.HeUniform),
    tf.keras.layers.BatchNormalization(axis=3),
    tf.keras.layers.PReLU(shared_axes=[1, 2]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.ReLU(),
    # layers.Dropout(0.7),
    tf.keras.layers.Dense(256),
    tf.keras.layers.Softmax()])

temp_dataset = get_local_dataset(lidar_training_path, beam_training_path,NUM_VEHICLES,0)
preprocessed_example_dataset=preprocess(temp_dataset)
example_element = next(iter((preprocessed_example_dataset)))

def model_fn():
  keras_model = create_keras_model()
  top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
  top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)
  return tff.learning.from_keras_model(keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[top1,top10])


################################Main################################   
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr=5e-3),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(lr=.25))

evaluation = tff.learning.build_federated_evaluation(model_fn)

test_data = LidarDataset2D(lidar_test_path, beam_test_path)
test_data.lidar_data = np.transpose(test_data.lidar_data, (0, 2, 3, 1))
    
accFL=0  
for MONTECARLOi in range(MONTECARLO):
    ###Generate Federated Train Dataset
    federated_train_data=[]     
    for i in range(NUM_VEHICLES):
        train_dataset = get_local_dataset(lidar_training_path, beam_training_path,NUM_VEHICLES,i)
        federated_train_data.append(preprocess(train_dataset))
        
    ###Generate Test Dataset
    test_dataset = get_test_dataset(lidar_test_path, beam_test_path)
    federated_test_data=[preprocess(test_dataset)]
    
    top1=np.zeros(AGGREGATION_ROUNDS)
    top10=np.zeros(AGGREGATION_ROUNDS)
    
    state = iterative_process.initialize() ###Initialize training
    
    #top1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_categorical_accuracy', dtype=None)
    #top10 = tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10_categorical_accuracy', dtype=None)
    
    ###Federated Training    
    for round_num in range(AGGREGATION_ROUNDS):
      state, metrics = iterative_process.next(state, federated_train_data)
      test_metrics = evaluation(state.model, federated_test_data)
      
      print(str(metrics))
      print(str(test_metrics))
      
      top1[round_num]=test_metrics['top_1_categorical_accuracy']
      top10[round_num]=test_metrics['top_10_categorical_accuracy']
    
      ###Generate Accuracy and Throughput Performance Curves
      keras_model = create_keras_model()
      #keras_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[top1,top10])
      state.model.assign_weights_to(keras_model)
      test_preds = keras_model.predict(test_data.lidar_data, batch_size=100)
      test_preds_idx = np.argsort(test_preds, axis=1)
      top_k = np.zeros(100)
      throughput_ratio_at_k = np.zeros(100)
      correct = 0
      for i in range(100):
        correct += np.sum(test_preds_idx[:, -1-i] == np.argmax(test_data.beam_output, axis=1))
        top_k[i] = correct/test_data.beam_output.shape[0]
        throughput_ratio_at_k[i] = np.sum(np.log2(np.max(np.take_along_axis(test_data.beam_output_true, test_preds_idx, axis=1)[:, -1-i:], axis=1) + 1.0))/\
                                   np.sum(np.log2(np.max(test_data.beam_output_true, axis=1) + 1.0))
        
      sio.savemat('federated_accuracy'+str(round_num)+'.mat',{'accuracy':top_k})
      sio.savemat('federated_throughput'+str(round_num)+'.mat',{'throughput':throughput_ratio_at_k})
    
    sio.savemat('top1.mat',{'top1':top1})
    sio.savemat('top10.mat',{'top10':top10})
    
    np.savez("federated.npz", classification=top_k, throughput_ratio=throughput_ratio_at_k)
    accFL=accFL+metrics['train']['top_10_categorical_accuracy']/MONTECARLO
    
    print(MONTECARLOi)
    
print(accFL)