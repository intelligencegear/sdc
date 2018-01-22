import keras
import cv2
import random
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.layers import Input,Dense,Flatten,Lambda,Conv2D,Dropout,Cropping2D,Activation,MaxPooling2D
from keras.models import Model


##################################################################################
## Super parameters
##################################################################################
TEST_SPLIT=0.2
BATCH_SIZE=32
MAX_EPOCHS=20
SAMPLES_PER_EPOCH=20000
CONV_INITIALIZER='normal'
FC_INITIALIZER='lecun_uniform'
DROPOUT_PROB=0.5

NVIDIA_CORRECT=0.25 ## for nvidia model
LENET5_CORRECT=0.5  ## for lenet model
AUGMENT_PROB=0.5
HORIZONTAL_SHIFT_PER_PIXEL=0.002
TRANSLATE_H_RANGE=100
TRANSLATE_V_RANGE=50

IMG_W, IMG_H, IMG_C = 320, 160, 3
INPUT_SHAPE = (IMG_H, IMG_W, IMG_C)
CROPPING_SHAPE=((60,20), (0,0))

##################################################################################
## Data augmentation related utils
##################################################################################

# load RGB image
def load_image(image_file):
    image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    return image

# converting from rgb to yuv color space, just as what Nvidia paper does
def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# center-left-right random augmentation
def clr_random(center, left, right, steering, correct):
    if correct == 0:
        return load_image(center), steering
    else:
        choice = random.choice('clr')
        if choice == 'l':
            return load_image(left), steering + correct
        elif choice == 'r':
            return load_image(right), steering - correct
        else:
            return load_image(center), steering

# flipping random augmentation
def flip_random(image, steering):
    if np.random.uniform() < 0.5:
        image = np.fliplr(image)
        steering = -steering
    
    return image, steering

# translation random augmentation
def translate_random(image, steering):
    trans_h = TRANSLATE_H_RANGE * (np.random.uniform() - 0.5)
    trans_v = TRANSLATE_V_RANGE * (np.random.uniform() - 0.5)
    steering = steering + trans_h * HORIZONTAL_SHIFT_PER_PIXEL
    
    trans_m = np.float32([[1, 0, trans_h], [0, 1, trans_v]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    
    return image, steering

# shadow random augmentation, assuming the image is rgb color space
def shadow_random(image):
    l = np.random.uniform(low=0, high=0.5)
    
    x1, y1 = IMG_W * np.random.rand(), 0
    x2, y2 = IMG_W * np.random.rand(), IMG_H
    xm, ym = np.mgrid[0:IMG_H, 0:IMG_W]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = (mask == np.random.randint(2))    
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls_image[:, :, 1][cond] = hls_image[:, :, 1][cond] * l
    
    return cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)

# brightness random augmentation, assuming the image is rgb color space
def brightness_random(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.5 + (np.random.uniform() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * random_bright
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# all augmentation processing
def augment(center, left, right, steering, correct):
    image, steering = clr_random(center, left, right, steering, correct)
    image, steering = flip_random(image, steering)
    image, steering = translate_random(image, steering)
    image = shadow_random(image)
    image = brightness_random(image)
    
    return image, steering

# data generator
def generator(data, batch_size, model, process):
    
    while 1:
        samples = shuffle(data)
        num_samples = len(samples)
        
        print("%s: traing sampling %d data." % (process, num_samples))
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images=[]
            steerings=[]
            
            for batch_sample in batch_samples:
                center, left, right = batch_sample[0], batch_sample[1], batch_sample[2]
                steering = batch_sample[3]
                
                correct = 0
                if model == 'nvidia':
                    correct = NVIDIA_CORRECT
                elif model == 'lenet5':
                    correct = LENET5_CORRECT
                else:
                    correct = 0
                
                # argumentation only when traing
                if process=='train' and random.random() < AUGMENT_PROB:
                    image, steering = augment(center, left, right, steering, correct)
                else:
                    image = load_image(center)
                
                # rgb->yuv
                image = rgb2yuv(image)
                images.append(image)
                steerings.append(steering)

            
            X_train = np.array(images)
            y_train = np.array(steerings)
            
            yield shuffle(X_train, y_train)


##################################################################################
## Nvidia End2End Model
##################################################################################

def nvidia_resize_op(image):
    from keras.backend import tf as ktf   
    image = ktf.image.resize_images(image, (66, 200))
    return image

def build_nvidia_model(shape=INPUT_SHAPE):
    inputs=Input(shape, name='input')
    
    ## cropping operation    
    x = Cropping2D(cropping=CROPPING_SHAPE, name='cropping')(inputs)
    
    ## nomalizing operation
    x = Lambda(lambda x: x/127.5 - 1, name='normalizing')(x)
    
    ## resizing operation
    x = Lambda(nvidia_resize_op, name='resizing')(x)
    
    ## convolution Layers
    x = Conv2D(24,5,5, activation='elu', subsample=(2,2), name='conv1', init=CONV_INITIALIZER)(x)    
    x = Conv2D(36,5,5, activation='elu', subsample=(2,2), name='conv2', init=CONV_INITIALIZER)(x)
    x = Conv2D(48,5,5, activation='elu', subsample=(2,2), name='conv3', init=CONV_INITIALIZER)(x)    
    x = Conv2D(64,3,3, activation='elu', subsample=(1,1), name='conv4', init=CONV_INITIALIZER)(x)
    x = Conv2D(64,3,3, activation='elu', subsample=(1,1), name='conv5', init=CONV_INITIALIZER)(x)  
    
    ## fully connected layers
    x = Flatten()(x)
    x = Dropout(DROPOUT_PROB, name='dropout_1')(x)
    x = Dense(100, activation='elu', name='fc1', init=FC_INITIALIZER)(x)
#     x = Dropout(DROPOUT_PROB, name='dropout_2')(x)
    x = Dense(50, activation='elu', name='fc2', init=FC_INITIALIZER)(x)
#     x = Dropout(DROPOUT_PROB, name='dropout_3')(x)
    x = Dense(10, activation='elu', name='fc3', init=FC_INITIALIZER)(x)
    y = Dense(1, name='output')(x)

    ## model
    model = Model(input=inputs, output=y, name='nvidia_end2end_model')
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    model.summary()

    return model

def train_nvida_model(model, data, save_path='models/nvidia_model_v5-{epoch:03d}.h5'):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=10, 
        verbose=1, 
        mode='auto'
    )

    save_best_model = keras.callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='auto'
    )
    
    train_data, validation_data = train_test_split(data, test_size=TEST_SPLIT, random_state=0)

    history = model.fit_generator(
        generator(train_data, BATCH_SIZE, 'nvidia', 'train'),
        nb_epoch=MAX_EPOCHS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_data=generator(validation_data, BATCH_SIZE, 'nvidia', 'validation'),
        nb_val_samples=len(validation_data),
        callbacks=[early_stopping, save_best_model],
        verbose=1
    )
    
    return history


##################################################################################
## Lenet5 Model
##################################################################################

def lenet5_resize_op(image):
    from keras.backend import tf as ktf   
    image = ktf.image.resize_images(image, (32, 32))
    return image

def build_lenet5_model(shape=(160,320,3)):
    inputs=Input(shape, name='input')
    
    ## cropping operation    
    x = Cropping2D(cropping=CROPPING_SHAPE, name='cropping')(inputs)
    
    ## nomalizing operation
    x = Lambda(lambda x: x/127.5 - 1, name='normalizing')(x)
    
    ## resizing operation
    x = Lambda(lenet5_resize_op, name='resizing')(x)
    
    ## convolution layers
    x = Conv2D(6,5,5, subsample=(1,1), border_mode='valid', name='conv1', init=CONV_INITIALIZER)(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Activation('relu')(x)
    x = Conv2D(16,5,5, subsample=(1,1), border_mode='valid', name='conv2', init=CONV_INITIALIZER)(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)
    x = Activation('relu')(x)
    
    ## fully connected layers
    x = Flatten()(x)
    x = Dense(120, activation='relu', name='fc1', init=FC_INITIALIZER)(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_PROB)(x)
    x = Dense(84, activation='relu', name='fc2', init=FC_INITIALIZER)(x)
    x = Activation('relu')(x)
    x = Dropout(DROPOUT_PROB)(x)
    y = Dense(1, name='output', init=FC_INITIALIZER)(x)
    
    ## model
    model = Model(input=inputs, output=y, name='lenet_5_model')
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    return model

def train_lenet5_model(model, data, save_path='models/lenet5_model_v5-{epoch:03d}.h5'):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=10, 
        verbose=1, 
        mode='auto'
    )

    save_best_model = keras.callbacks.ModelCheckpoint(
        save_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='auto'
    )
    
    train_data, validation_data = train_test_split(data, test_size=TEST_SPLIT, random_state=0)

    history = model.fit_generator(
        generator(train_data, BATCH_SIZE, 'lenet5', 'train'),
        nb_epoch=MAX_EPOCHS,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        validation_data=generator(validation_data, BATCH_SIZE, 'lenet5', 'validation'),
        nb_val_samples=len(validation_data),
        callbacks=[early_stopping, save_best_model],
        verbose=1
    )
    
    return history