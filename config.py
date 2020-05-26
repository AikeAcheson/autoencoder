import os

train = '/Users/chyson/hx/data/kaggle-stain/train'
train_cleaned = '/Users/chyson/hx/data/kaggle-stain/train_cleaned'
# test = '/Users/chyson/hx/data/hx/hx/data/origin'
test = '/Users/chyson/hx/data/kaggle-stain/test'

mode = 'train'

img_row = 320
img_col = 320
channels = 1
img_shape = (img_row, img_col, channels)

checkpoints = 'checkpoints'
if not os.path.exists(checkpoints):
    os.mkdir(checkpoints)
model = ''
start_epoch = 0

fig_path = 'monitor.png'
json_path = 'monitor.json'
