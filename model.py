import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


def build_model():
    # Deep model
    # Build Deep Learning Model

    model = Sequential()

    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    return model


def train_model(model, train_data, val_data):
    # Train model
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

    return hist, model


def evaluate_model(model, test_data):
    # Evaluate Performance

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test_data.as_numpy_iterator():
        X, y = batch
        y_pred = model.predict(X)
        pre.update_state(y, y_pred)
        re.update_state(y, y_pred)
        acc.update_state(y, y_pred)

    print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')


def test_model(model, image_path):
    # Test
    img = cv2.imread(image_path)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.imshow()

    resize = tf.image.resize(img, (256, 256))
    # plt.imshow(resize.numpy().astype(int))
    # plt.show()
    y_pred = model.predict(np.expand_dims(resize / 255, 0))

    if y_pred > 0.5:
        print(f"{image_path} predicted class is Sad")
        return "sad"
    else:
        print(f"{image_path} predicted class is Happy")
        return "happy"


def test_model_dataset(model, test_set):
    for image_class in os.listdir(test_set):
        print(f"Test for {image_class} class:")
        print(f"-----------------------------")
        number_of_img = os.listdir(os.path.join(test_set, image_class))
        print(number_of_img)
        correct_tests = 0
        for image in number_of_img:
            image_path = os.path.join(test_set, image_class, image)
            test_result = test_model(model, os.path.join(test_set, image_class, image))
            if test_result == image_class:
                correct_tests = correct_tests + 1
        print(f"Number of correct tests {correct_tests} from a total number of {len(number_of_img)} images")
        print("")


def save_model(model, model_name):
    model_path = os.path.join('models', model_name)
    model.save(model_path)
    print(f"Model saved at path: {model_path}")


def load_model_from_path(model_name):
    print(f"Model loaded from path: {model_name}")
    model = load_model(os.path.join('models', model_name))

    return model
