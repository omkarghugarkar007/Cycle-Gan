import tensorflow as tf
import tensorflow_datasets as tfds

data, metadata = tfds.load('cycle_gan/monet2photo', with_info=True, as_supervised=True)
train_x, train_y, test_x, test_y = data['trainA'], data['trainB'], data['testA'], data['testB']

img_rows = 256
img_cols = 256
channels = 3


def preprocess_img(image,_):

    return tf.reshape(tf.cast(tf.image.resize(image, (int(img_rows),int(img_cols))),tf.float32)/127.5-1,(1,img_rows,img_cols,channels))

train_x = train_x.map(preprocess_img)
train_y = train_y.map(preprocess_img)
test_x = test_x.map(preprocess_img)
test_y = test_y.map(preprocess_img)

