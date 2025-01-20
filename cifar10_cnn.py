# Author: Matthew Alderman
# Date: 07/27/2024


import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert label vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Split training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_val, x_test, y_train, y_val, y_test


def create_model():
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    # First convolutional block
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def create_model_with_bn():
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and fully connected layer
    model.add(Flatten())
    model.add(Dense(512, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


def compile_and_train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs, learning_rate, checkpoint_path):
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
                        callbacks=[checkpoint])
    return history


def evaluate_model(model, x_train, y_train, x_val, y_val):
    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    print(f'Training loss: {train_loss}, Training accuracy: {train_accuracy}')
    print(f'Validation loss: {val_loss}, Validation accuracy: {val_accuracy}')
    return train_loss, train_accuracy, val_loss, val_accuracy


def plot_loss(history, title):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_with_data_augmentation(x_train, y_train, x_val, y_val, model, batch_size, epochs, learning_rate,
                                 checkpoint_path):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min')
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                        validation_data=(x_val, y_val), callbacks=[checkpoint])
    return history


if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_preprocess_data()

    # Model without data augmentation
    model = create_model()
    history = compile_and_train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=50,
                                      learning_rate=0.001, checkpoint_path='best_model.keras')
    evaluate_model(model, x_train, y_train, x_val, y_val)
    plot_loss(history, 'Training and Validation Loss')

    # Model with data augmentation
    model_aug = create_model()
    history_aug = train_with_data_augmentation(x_train, y_train, x_val, y_val, model_aug, batch_size=32, epochs=50,
                                               learning_rate=0.001, checkpoint_path='best_model_aug.keras')
    evaluate_model(model_aug, x_train, y_train, x_val, y_val)
    plot_loss(history_aug, 'Training and Validation Loss with Data Augmentation')

    # Model with batch normalization
    model_bn = create_model_with_bn()
    history_bn = compile_and_train_model(model_bn, x_train, y_train, x_val, y_val, batch_size=64, epochs=50,
                                         learning_rate=0.01, checkpoint_path='best_model_bn.keras')
    evaluate_model(model_bn, x_train, y_train, x_val, y_val)
    plot_loss(history_bn, 'Training and Validation Loss with Batch Normalization')
