from tensorflow.keras import layers, models, initializers

Lidar2D = models.Sequential([
    layers.Input(shape=(20, 200, 1)),
    layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Conv2D(5, 3, 1, padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Conv2D(5, 3, 2, padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Conv2D(1, 3, (1, 2), padding='same', kernel_initializer=initializers.HeUniform),
    layers.BatchNormalization(axis=3),
    layers.PReLU(shared_axes=[1, 2]),
    layers.Flatten(),
    layers.Dense(16),
    layers.ReLU(),
    # layers.Dropout(0.7),
    layers.Dense(256),
    layers.Softmax()
])

LidarMarcus = models.Sequential([
    layers.Input(shape=(20, 200, 10)),
    layers.Conv2D(10, kernel_size=(13, 13),
                   activation='relu',
                   padding="same"),
    layers.Conv2D(30, (11, 11), padding="SAME", activation='relu'),
    layers.Conv2D(25, (9, 9), padding="SAME", activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 1)),
    layers.Dropout(0.3),
    layers.Conv2D(20, (7, 7), padding="SAME", activation='relu'),
    layers.MaxPooling2D(pool_size=(1, 2)),
    layers.Conv2D(15, (5, 5), padding="SAME", activation='relu'),
    layers.Dropout(0.3),
    layers.Conv2D(10, (3, 3), padding="SAME", activation='relu'),
    layers.Conv2D(1, (1, 1), padding="SAME", activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='softmax')
])


if __name__ == '__main__':
    model = Lidar2D
    model.summary()
