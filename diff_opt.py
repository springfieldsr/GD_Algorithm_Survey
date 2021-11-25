import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers, optimizers
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def cnn_classifier(opt):
        config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 1}) 
        sess = tf.Session(config=config) 
        keras.backend.set_session(sess)

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        #z-score
        mean = np.mean(x_train,axis=(0,1,2,3))
        std = np.std(x_train,axis=(0,1,2,3))
        x_train = (x_train-mean)/(std+1e-7)
        x_test = (x_test-mean)/(std+1e-7)

        num_classes = 10
        y_train = np_utils.to_categorical(y_train,num_classes)
        y_test = np_utils.to_categorical(y_test,num_classes)

        baseMapNum = 32
        weight_decay = 1e-4
        model = Sequential()
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(4*baseMapNum, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()

        #data augmentation
        datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
        )
        datagen.fit(x_train)

        #training
        batch_size = 64
        epochs=100
        model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test))
        model.save_weights('cifar10_normal_rms_ep75.h5')

        scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
        print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))
        keras.backend.clear_session()
        return history


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

history_list = []
history_list.append(cnn_classifier(keras.optimizers.adam(lr=0.001,decay=1e-6)))
history_list.append(cnn_classifier(keras.optimizers.sgd(lr=0.001,decay=1e-6)))
history_list.append(cnn_classifier(keras.optimizers.adagrad(lr=0.001,decay=1e-6)))
history_list.append(cnn_classifier(keras.optimizers.rmsprop(lr=0.001,decay=1e-6)))

opt_name = ['adam', 'sgd', 'adagrad', 'rmsprop']
ax = plt.gca()
for i in range(len(history_list)):
        history = history_list[i]
        name = opt_name[i]
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(history.history['acc'], label=name, color=color)
        plt.plot(history.history['val_acc'], '--', color=color)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
plt.legend()
plt.show()
for i in range(len(history_list)):
        history = history_list[i]
        name = opt_name[i]
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(history.history['loss'], label=name, color=color)
        plt.plot(history.history['val_loss'], '--', color=color)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
plt.legend()
plt.show()