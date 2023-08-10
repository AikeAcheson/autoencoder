import config
from imutils import paths

from sklearn.model_selection import train_test_split
from autoencoder import AutoEncoder
from utils import load_image

from hx.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from hx.preprocessing.simple_preprocessor import SimplePreprocessor
from hx.preprocessing.zero_one_preprocessor import ZeroOnePreprocessor
from hx.datasets.simple_image_loader import SimpleImageLoader


# list of images
train_img_paths = list(paths.list_images(config.train))
cleaned_img_paths = list(paths.list_images(config.train_cleaned))
test_img_paths = list(paths.list_images(config.test))


# load data
train_data = load_image(train_img_paths)
cleaned_data = load_image(cleaned_img_paths)
test_data = load_image(test_img_paths)
print(train_data.shape, cleaned_data.shape, test_data.shape)

# train validate split
x_train, x_val, y_train, y_val = train_test_split(train_data, cleaned_data, train_size=0.8, random_state=42)
print(x_train.shape, x_val.shape)

# model
ae = AutoEncoder()

if config.mode == 'train':
    ae.train_model(x_train, y_train, x_val, y_val, epochs=7, batch_size=10)





