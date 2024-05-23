import os
from m7016h import default_parameters as dp

from PIL import Image

if dp.is_mac:
    from keras_preprocessing.image import ImageDataGenerator
else:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from m7016h.default_parameters import absolute_dataset_path


def initialize_generators(train_dir=dp.train_dir, val_dir=dp.val_dir, test_dir=dp.test_dir):
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator()
    image_size = ()
    for files in os.walk(os.path.join(absolute_dataset_path(train_dir), 'benign')):
        image_path = os.path.join(absolute_dataset_path(train_dir), 'benign', files[2][0])
        print(f"{image_path}")
        with Image.open(image_path) as img:
            image_size = img.size
        break

    target_image_width = image_size[0] // dp.image_size_divider
    target_image_height = image_size[1] // dp.image_size_divider

    train_generator = train_datagen.flow_from_directory(
        absolute_dataset_path(train_dir),
        target_size=(target_image_width, target_image_height),
        batch_size=dp.batch_size,
        class_mode='binary'
    )

    val_generator = val_test_datagen.flow_from_directory(
        absolute_dataset_path(val_dir),
        target_size=(target_image_width, target_image_height),
        batch_size=dp.batch_size,
        class_mode='binary'
    )

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        absolute_dataset_path(test_dir),
        target_size=(target_image_width, target_image_height),
        batch_size=dp.batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_datagen, train_generator, val_generator, test_generator, (target_image_width, target_image_height)
