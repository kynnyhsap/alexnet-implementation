from tensorflow.keras import Sequential, layers


def build_alexnet(input_width=224, input_height=224, input_channels=3, num_classes=1000):
    return Sequential([
        # Layer 1
        layers.Conv2D(input_shape=(input_width, input_height, input_channels),
                      filters=96, kernel_size=11, strides=4, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),

        # Layer 2
        layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),

        # Layer 3
        layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),

        # Layer 4
        layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),

        # Layer 5
        layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPool2D(pool_size=3, strides=2),

        layers.Flatten(),

        # Layer 6
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        # Layer 7
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        # Layer 8
        layers.Dense(num_classes, activation='softmax')
    ])
