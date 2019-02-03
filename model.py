from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 300, 3), data_format = 'channels_first'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

# model.add(Conv2D(32, (3, 3), input_shape=(3, 100, 300), data_format = 'channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

# model.add(Conv2D(64, (3, 3), input_shape=(3, 100, 300), data_format = 'channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

imggen = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest',
    rescale = 1./255,
    shear_range = 0.25,
    zoom_range = 0.25
)

train_normal_dir = 'Train/Normal'
train_white_dir = 'Train/White_Spot'
val_normal_dir = 'Validation/Normal'
val_white_dir = 'Validation/White_Spot'
train_dir = 'Train'
val_dir = 'Validation'

batch_size = 16

train_generator = imggen.flow_from_directory(directory = train_dir,
                                             target_size = (100, 300),
                                             batch_size = batch_size,
                                             class_mode = 'binary')

validation_generator = imggen.flow_from_directory(directory = val_dir,
                                                  target_size = (100, 300),
                                                  batch_size = batch_size,
                                                  class_mode = 'binary')

model.fit_generator(generator = train_generator,
                    steps_per_epoch = 2000 // batch_size,
                    epochs = 5,
                    validation_data = validation_generator,
                    validation_steps = 800 // batch_size)
model.save_weights('try_one.h5')
