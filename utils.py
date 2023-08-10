import numpy as np
from keras.preprocessing import image
import config


def load_image(path):
    image_list = np.zeros((len(path), config.width, config.height, 1))
    for i, fig in enumerate(path):
        img = image.load_img(fig, color_mode='grayscale', target_size=(config.width, config.height))
        x = image.img_to_array(img).astype('float32')
        x = x / 255.0
        image_list[i] = x

    return image_list
