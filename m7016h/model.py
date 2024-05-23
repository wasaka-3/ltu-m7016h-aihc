import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model


def create_model(target_image_width, target_image_height):
    # Build the CNN model
    model = Sequential([
        Conv2D(filters=64, kernel_size=(5, 5), activation='relu',
               input_shape=(target_image_width, target_image_height, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Conv2D(filters=80, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),
        Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),  # What about removing the dropout layer?
        Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.Precision()
        ])
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    return model
