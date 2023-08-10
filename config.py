import os

train = '/Users/chyson/hx/data/kaggle-stain/train'
train_cleaned = '/Users/chyson/hx/data/kaggle-stain/train_cleaned'
# test = '/Users/chyson/hx/data/hx/hx/data/origin'
test = '/Users/chyson/hx/data/kaggle-stain/test'

mode = 'train'

width = 320
height = 320
# img_row = 258
# img_col = 540
channels = 1
shape = (width, height, channels)

checkpoints = 'checkpoints'
if not os.path.exists(checkpoints):
    os.mkdir(checkpoints)
model = ''
start_epoch = 3

fig_path = 'monitor.png'
json_path = 'monitor.json'

index = 3
