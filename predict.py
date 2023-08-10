import config
from imutils import paths
import cv2
import matplotlib.pyplot as plt
from utils import load_image
from autoencoder import AutoEncoder

from hx.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from hx.preprocessing.simple_preprocessor import SimplePreprocessor
from hx.preprocessing.zero_one_preprocessor import ZeroOnePreprocessor
from hx.datasets.simple_image_loader import SimpleImageLoader

test_img_paths = list(paths.list_images(config.test))



# load data
test_data = load_image(test_img_paths)

# load model
ae = AutoEncoder()
ae.load_model(5)

# predict
preds = ae.model.predict(test_data)
# for i in range(len(preds)):
#     pred = preds[i] * 255.0
#     pred = pred.reshape(config.width,config.height)
#     plt.imshow(pred,cmap='gray')
#     plt.show()
preds_0 = preds[10] * 255.0
preds_0 = preds_0.reshape(config.width, config.height)
x_test_0 = test_data[10] * 255.0
x_test_0 = x_test_0.reshape(config.width, config.height)
plt.imshow(x_test_0, cmap='gray')
plt.imshow(preds_0,cmap='gray')
plt.show()


