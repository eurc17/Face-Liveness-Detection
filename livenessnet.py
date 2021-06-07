# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

class LivenessNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
  
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
  
		base_model = Xception(weights='imagenet', include_top=False, input_shape=inputShape)
		x = base_model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(64, activation='relu')(x)
		predictions = Dense(classes, activation='softmax')(x)
		model = Model(inputs=base_model.input, outputs=predictions)


		# return the constructed network architecture
		return model