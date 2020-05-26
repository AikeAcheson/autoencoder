import config
from imutils import paths

from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder

from hx.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from hx.preprocessing.simple_preprocessor import SimplePreprocessor
from hx.preprocessing.zero_one_preprocessor import ZeroOnePreprocessor
from hx.datasets.simple_image_loader import SimpleImageLoader


# list of images
train_img_paths = list(paths.list_images(config.train))
cleaned_img_paths = list(paths.list_images(config.train_cleaned))
test_img_paths = list(paths.list_images(config.test))

# initialize the image preprocessors
sp = SimplePreprocessor(320, 320)
iap = ImageToArrayPreprocessor()
zop = ZeroOnePreprocessor()

# load data
sil = SimpleImageLoader(preprocessors=[sp, iap, zop], graystyle=True)
train_data = sil.load(train_img_paths, verbose=50)
cleaned_data = sil.load(cleaned_img_paths, verbose=50)
test_data = sil.load(test_img_paths, verbose=50)
print(train_data.shape, cleaned_data.shape, test_data.shape)

# train validate split
x_train, x_val, y_train, y_val = train_test_split(train_data, cleaned_data, train_size=0.1, random_state=42)
print(x_train.shape, x_val.shape)

# model
ae = Autoencoder()

if config.mode == 'train':
    ae.train_model(x_train, y_train, x_val, y_val, epochs=7, batch_size=10)
else:
    pass
