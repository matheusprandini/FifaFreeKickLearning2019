from __future__ import division, print_function
import collections
import numpy as np
import cv2
import os
from keras.models import *
from keras.initializers import *
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import *
from scipy.misc import imresize, imsave
import tensorflow as tf

# Initialize Global Parameters
DATA_DIR = "Models/"
MEMORY_SIZE = 10000 # number of previous transitions to remember
IMAGE_SIZE = 84

class DQN:

    def __init__(self, num_actions):
        self.model = self.build_model(num_actions)
        self.target_model = None
        self.experience = collections.deque(maxlen=MEMORY_SIZE)
	
    # Load Model (Main DQN)
    def load_model(self, file_name, huber_loss=True):
        if huber_loss:
            self.model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")), custom_objects={'huber_loss': self.huber_loss})
        else:
            self.model = load_model(os.path.join(DATA_DIR, (file_name + ".h5")))
		
    # Save Model (Main DQN)
    def save_model(self, file_name):
        self.model.save(os.path.join(DATA_DIR, (file_name + ".h5")), overwrite=True)
		
    # Save DQN model and update Target Model
    def update_target_network(self):
        self.save_model("target_network")
        self.target_model = load_model(os.path.join(DATA_DIR, ("target_network.h5")), custom_objects={'huber_loss': self.huber_loss})

    # build the model
    def build_model(self, num_actions):
	
		# Sequential Model
        model = Sequential()
		
		# 1st cnn layer
        model.add(Conv2D(32, kernel_size=8, strides=4, 
                 kernel_initializer=VarianceScaling(), 
                 padding="same",
                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 4)))
        model.add(Activation("relu"))
		
        # 2st cnn layer
        model.add(Conv2D(64, kernel_size=4, strides=2, 
                 kernel_initializer=VarianceScaling(), 
                 padding="same"))
        model.add(Activation("relu"))
		
		# 3st cnn layer
        model.add(Conv2D(64, kernel_size=3, strides=1,
                 kernel_initializer=VarianceScaling(),
                 padding="same"))
        model.add(Activation("relu"))
		
		# Flattening parameters
        model.add(Flatten())
		
		# 1st mlp layer
        model.add(Dense(512, kernel_initializer=VarianceScaling()))
        model.add(Activation("relu"))
		
		# 2st (last) cnn layer -> Classification layer (left, right, low_shoot, high_shoot)
        model.add(Dense(num_actions, kernel_initializer=VarianceScaling()))
		
		# Compiling Model
        model.compile(optimizer=RMSprop(lr=1e-4), loss=self.huber_loss)

		# Show model details
        model.summary()
		
        return model

	# Defining the huber loss
    def huber_loss(self, y_true, y_pred):
        return tf.losses.huber_loss(y_true,y_pred)

    # Preprocess images and stacks in a deque
    def preprocess_images(self,images):

        if images.shape[0] < 4:
            # single image
            x_t = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
            x_t = imresize(x_t, (IMAGE_SIZE, IMAGE_SIZE))
            x_t = x_t.astype("float")
            x_t /= 255.0
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        else:
            # 4 images
            xt_list = []
            for i in range(images.shape[0]):
                x_t = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                x_t = imresize(x_t, (IMAGE_SIZE, IMAGE_SIZE))
                x_t = x_t.astype("float")
                x_t /= 255.0
                xt_list.append(x_t)
                #imsave('outfile' + str(i) + '.jpg', x_t)
            s_t = np.stack((xt_list[0], xt_list[1], xt_list[2], xt_list[3]), 
                       axis=2)
        s_t = np.expand_dims(s_t, axis=0)

        return s_t
	
	# Return a batch of experiencie to train the dqn model
    def get_next_batch_target(self, num_actions, gamma, batch_size):
        batch_indices = np.random.randint(low=0, high=len(self.experience),
                                      size=batch_size)
        batch = [self.experience[i] for i in batch_indices]
        X = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 4))
        Y = np.zeros((batch_size, num_actions))
		
        # Building the batch data
        for i in range(len(batch)):
            s_t, a_t, r_t, s_tp1, game_over = batch[i]
            X[i] = s_t
            Y[i] = self.target_model.predict(s_t)[0]
            Q_sa = np.max(self.target_model.predict(s_tp1)[0])
            if game_over:
                Y[i, a_t] = r_t
            else:
                Y[i, a_t] = r_t + gamma * Q_sa

        return X, Y
		
    def get_next_batch(self, num_actions, gamma, batch_size):
        batch_indices = np.random.randint(low=0, high=len(self.experience),
                                      size=batch_size)
        batch = [self.experience[i] for i in batch_indices]
        X = np.zeros((batch_size, IMAGE_SIZE, IMAGE_SIZE, 4))
        Y = np.zeros((batch_size, num_actions))
		
        # Building the batch data
        for i in range(len(batch)):
            s_t, a_t, r_t, s_tp1, game_over = batch[i]
            X[i] = s_t
            Y[i] = self.model.predict(s_t)[0]
            Q_sa = np.max(self.model.predict(s_tp1)[0])
            if game_over:
                Y[i, a_t] = r_t
            else:
                Y[i, a_t] = r_t + (gamma * Q_sa)

        return X, Y
        
print("")
		
if __name__ == '__main__':
	
    dqn = DQN()