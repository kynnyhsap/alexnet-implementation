import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from alexnet import build_alexnet
from dataset import load_dataset, CLASS_NAMES

MODEL_FILE_PATH = './model/AlexNet.h5'

INPUT_WIDTH = 150
INPUT_HEIGHT = 150
INPUT_CHANNELS = 3

NUM_CLASSES = 6

EPOCHS = 20
BATCH_SIZE = 128

print('Creating model.')
model = build_alexnet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)

print('Compiling model.')
optimizer = optimizers.Adam()
loss = losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.summary()

print('Loading dataset...')
x, y = load_dataset(train=True, one_hot=True, shuffle=False)

print('Dataset loaded: ')
print('x shape:', x.shape)
print('y shape:', y.shape)

print('Start training: ')
model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05, verbose=2)
print('Finished training.')

model.save(filepath=MODEL_FILE_PATH)
print(f'Model saved to "{MODEL_FILE_PATH}"')
