import cv2
import os
import numpy as np

DATA_FOLDER = './data/'

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


def load_image_from_folder(folder_path):
    images = []

    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))

        if img is not None:
            if img.shape != (150, 150, 3):
                continue

            img = np.expand_dims(img, axis=0)
            images.append(img)

    return images


def load_dataset(train=True, one_hot=True, shuffle=False):
    folder = DATA_FOLDER + ('train/' if train else 'test/')

    images = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_folder_path = folder + class_name
        class_images = load_image_from_folder(class_folder_path)
        class_labels = [label] * len(class_images)

        images += class_images
        labels += class_labels

    labels = np.array(labels)
    images = np.vstack(images)

    images = images / 255.

    if one_hot:
        m = len(labels)

        oh = np.zeros((m, labels.max() + 1))
        oh[np.arange(m), labels] = 1

        labels = oh

    if shuffle:
        indexes = np.arange(labels.shape[0])
        np.random.shuffle(indexes)

        images = images[indexes]
        labels = labels[indexes]

    return images, labels
