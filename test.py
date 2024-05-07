import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


IMG_HEIGHT = 256
IMG_WIDTH = 256


def normalize_test_image(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image


def resize_test_imgae(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image


def load_test_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    input_image = tf.cast(image, tf.float32)

    return input_image


def load_image_test(image_file):
    input_image = load_test_image(image_file)
    input_image = resize_test_imgae(input_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image = normalize_test_image(input_image)

    return input_image


test_dataset_test = tf.data.Dataset.list_files(
    'C:/Users/sufiyan/Desktop/pix2pix/content'+'/*.jpg')
test_dataset_test = test_dataset_test.map(load_image_test)


def herewego(model, test_input):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.show()


model = load_model(
    "C:/Users/sufiyan/Desktop/pix2pix/saved models/model_000145.h5")


for inp in test_dataset_test:
    inp = np.expand_dims(inp, axis=0)
    herewego(model, inp)
