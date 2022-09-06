from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    add,
    BatchNormalization,
    Bidirectional,
    concatenate,
    Conv1D,
    MaxPool1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    InputLayer,
    Lambda,
    LSTM,
    Reshape,
    TimeDistributed,
    LeakyReLU,
    ReLU,
)


import numpy as np


def create_model(Nr, Nt, T, layers, should_include_x=True, should_normalize=True):
    norm_factor = np.sqrt(Nr * Nt)

    input_shape = (T, ((Nr + Nt) if should_include_x else Nr) * 2)
    output_shape = (2 * Nr, Nt)
    numOutputs = np.prod(output_shape)

    model = Sequential()
    # model.add(InputLayer(input_shape=input_shape))
    model.add(Lambda(lambda x: x, input_shape=input_shape))
    for layer in layers:
        model.add(layer)

    model.add(Dense(numOutputs, activation="linear"))
    if True:
        model.add(Lambda(lambda x: norm_factor * l2_normalize(x, axis=-1)))
    model.add(Reshape(output_shape))

    return model


def time_dense_model(
    Nr: int,
    Nt: int,
    T: int,
    num_neurons_time: list,
    num_neurons: list,
    activation: str = "relu",
    should_include_x=True,
):
    layers = []
    for i, n in enumerate(num_neurons_time):
        layers.append(TimeDistributed(Dense(n, activation=activation)))

    layers.append(Flatten())

    for i, n in enumerate(num_neurons):
        layers.append(Dense(n, activation=activation))

    return create_model(Nr, Nt, T, layers, should_include_x=should_include_x)


def dense_model(
    Nr: int,
    Nt: int,
    T: int,
    num_neurons: list,
    activation: str = "relu",
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    should_include_x=True,
):

    layers = [Flatten()]

    for i, n in enumerate(num_neurons):
        layers.append(Dense(n, activation=activation))
        if use_batch_norm:
            layers.append(BatchNormalization())
        if dropout_rate != 0.0 and i != 0 and i != (len(num_neurons) - 1):
            layers.append(Dropout(dropout_rate))

    return create_model(Nr, Nt, T, layers, should_include_x=should_include_x)


def dense_leaky_model(
    Nr: int,
    Nt: int,
    T: int,
    num_neurons: list,
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    should_include_x=True,
):

    layers = [Flatten()]

    for i, n in enumerate(num_neurons):
        layers.append(Dense(n))
        layers.append(LeakyReLU())
        if use_batch_norm:
            layers.append(BatchNormalization())
        if dropout_rate != 0.0 and i != 0 and i != (len(num_neurons) - 1):
            layers.append(Dropout(dropout_rate))

    return create_model(Nr, Nt, T, layers, should_include_x=should_include_x)


def conv_lstm_model(
    Nr: int,
    Nt: int,
    T: int,
    convConfig: list,
    lstmConfig: list,
    lstm_last_return_sequence=True,
    activation: str = "relu",
    use_batch_norm: bool = False,
    dropout_rate: float = 0.0,
    should_include_x=True,
):

    layers = []

    for i, (filt, kern, stri) in enumerate(convConfig):
        layers.append(
            Conv1D(filt, kern, strides=stri, activation=activation, padding="same")
        )
        if use_batch_norm:
            layers.append(BatchNormalization())
        if dropout_rate != 0.0 and i != 0 and i != (len(convConfig) - 1):
            layers.append(Dropout(dropout_rate))

    for i, n in enumerate(lstmConfig):
        ret_seq = i < len(lstmConfig) - 1
        layers.append(LSTM(n, return_sequences=ret_seq))

    layers.append(Flatten())

    return create_model(Nr, Nt, T, layers, should_include_x=should_include_x)


def conv_res_model(Nr, Nt, T, should_include_x=False):

    input_shape = (T, ((Nr + Nt) if should_include_x else Nr) * 2)
    output_shape = (2 * Nr, Nt)
    numOutputs = np.prod(output_shape)
    inputs = Input(input_shape)

    x = Conv1D(64, 7, padding="same", strides=1)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    # x = MaxPool1D(pool_size=3, strides=2)(x)

    x1 = Conv1D(64, 3, padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(64, 3, padding="same")(x1)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = layers.add([x1, x])

    x2 = Conv1D(64, 3, padding="same")(x1)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = Conv1D(64, 3, padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)
    x2 = layers.add(x2, x1)

    xout = Flatten()(x2)
    xout = Dense(numOutputs, activation="linear")(xout)
    xout = Reshape(output_shape)(xout)
    model = Model(inputs=inputs, outputs=xout)
    return model


def ak_residual_model(Nr, Nt, T, activation="tanh", should_include_x=True):

    num_neurons = 10
    norm_factor = np.sqrt(Nr * Nt)

    input_shape = (T, ((Nr + Nt) if should_include_x else Nr) * 2)
    output_shape = (2 * Nr, Nt)
    numOutputs = np.prod(output_shape)
    inputs = Input(input_shape)

    x0 = Flatten()(inputs)

    d1 = Dense(2 * num_neurons, activation=activation)(x0)

    d2 = Dense(2 * num_neurons, activation=activation)(d1)

    r1 = Dense(1, activation=activation)(d2)

    z = add([x0, r1])

    d_out = Dense(numOutputs, activation="linear")(z)
    lambda1 = Lambda(lambda x: norm_factor * l2_normalize(x, axis=-1))(d_out)

    reshape1 = Reshape(output_shape)(lambda1)

    model = Model(inputs=inputs, outputs=reshape1)
    return model


def residual_model(Nr, Nt, T, num_neurons, activation="relu", should_include_x=True):
    norm_factor = np.sqrt(Nr * Nt)

    input_shape = (T, ((Nr + Nt) if should_include_x else Nr) * 2)
    output_shape = (2 * Nr, Nt)
    numOutputs = np.prod(output_shape)
    inputs = Input(input_shape)

    x0 = Flatten()(inputs)
    x1 = LeakyReLU()(Dense(num_neurons[0])(x0))
    for n in num_neurons[1:]:
        x0, x1 = x1, LeakyReLU()(Dense(n, activation=activation)(concatenate([x0, x1])))

    d_out = Dense(numOutputs, activation="linear")(concatenate([x0, x1]))

    # lambda1 = Lambda(lambda x: norm_factor * l2_normalize(x, axis=-1))(d_out)

    reshape1 = Reshape(output_shape)(d_out)

    model = Model(inputs=inputs, outputs=reshape1)

    return model


def toy_res_model(Nr, Nt, T, should_include_x=False):
    norm_factor = np.sqrt(Nr * Nt)
    input_shape = (T, ((Nr + Nt) if should_include_x else Nr) * 2)
    output_shape = (2 * Nr, Nt)
    numOutputs = np.prod(output_shape)
    inputs = Input(input_shape)

    x = layers.Conv1D(32, 3, activation="relu")(inputs)
    x = layers.Conv1D(64, 3, activation="relu")(x)
    b1_output = layers.MaxPooling1D(3)(x)

    x = layers.Conv1D(64, 3, activation="relu", padding="same")(b1_output)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    b2_output = layers.add([x, b1_output])

    x = layers.Conv1D(64, 3, activation="relu", padding="same")(b2_output)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    b3_output = layers.add([x, b2_output])

    x = layers.Conv1D(64, 3, activation="relu")(b3_output)
    # x = layers.GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    d1 = layers.Dense(256, activation="relu")(x)
    c1 = layers.concatenate([x, d1])
    d2 = layers.Dense(256, activation="relu")(c1)
    c2 = layers.concatenate([c1, d2])
    d3 = layers.Dense(256, activation="relu")(c2)
    c3 = layers.concatenate([c2, d3])
    # x = layers.Dropout(0.2)(x)
    x = layers.Dense(numOutputs)(c3)
    x = Lambda(lambda x: norm_factor * l2_normalize(x, axis=-1))(x)
    outputs = layers.Reshape(output_shape)(x)

    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":

    # dense_model(
    #    10, 10, 10, [20, 20, 20], use_batch_norm=True, dropout_rate=0.3
    # ).summary()

    # conv_lstm_model(
    #    10,
    #    10,
    #    20,
    #    [(100, 5, 1), (100, 3, 1), (100, 3, 1)],
    #    [10],
    #    dropout_rate=0.3,
    # ).summary()

    residual_model(10, 10, 10, [10, 10, 10]).summary()
