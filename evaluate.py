import tensorflow as tf
from dataset import load_dataset, CLASS_NAMES

print('Loading Model...')
alexnet = tf.keras.models.load_model('./model/AlexNet.h5')
print('Model Loaded.')

print('Loading test dataset...')
x, y = load_dataset(train=False, one_hot=True, shuffle=False)

print('Dataset loaded: ')
print('x shape:', x.shape)
print('y shape:', y.shape)

alexnet.evaluate(x, y)