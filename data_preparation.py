import cv2
import imghdr # check file extensions
import os
import tensorflow as tf


def remove_incorrect_images():
    data_dir = 'data'
    image_extensions = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                img_type = imghdr.what(image_path)
                if img_type not in image_extensions:
                    print("Image not in extension list {}".format(image_path))
                    os.remove(image_path)
            except Exception as e:
                print("Issue with image {}".format(image_path))


def load_data(path):
    data = tf.keras.utils.image_dataset_from_directory(path)
    data_iterator = data.as_numpy_iterator()
    # batch[0] -> class
    # batch[1] -> labels

    # class 1 = sad people
    # class 0 = happy people
    batch = data_iterator.next()

    return data, batch
