import keras
import numpy as np
from keras import backend as K
from keras import layers


def Model_AutoEncoder(Data, Target):
    Data = np.reshape(Data, (Data.shape[0], int(np.prod(Data.shape[1:]))))
    # This is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # Train_Data = Data.astype('float32') / 255.
    # Test_Data = Data.astype('float32') / 255.
    # Train_Data = Train_Data.reshape((len(Train_Data), np.prod(Train_Data.shape[1:])))
    # Test_Data = Test_Data.reshape((len(Test_Data), np.prod(Test_Data.shape[1:])))
    autoencoder.fit(Data, Target,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(Data, Target))
    encoded_imgs = encoder.predict(Data)
    pred = decoder.predict(encoded_imgs)

    inp = autoencoder.input  # input placeholder
    outputs = [layer.output for layer in autoencoder.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions

    layerNo = 1
    Feats = []
    for i in range(Data.shape[0]):
        test = Data[i, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    return Feats, pred


if __name__ == '_main_':
    a = np.random.random((200, 100, 3))
    b = np.random.randint(0, 2, (200, 3))
    c = np.random.random((200, 100, 3))
    d = np.random.randint(0, 2, (200, 1))
    Feats, pred = Model_AutoEncoder(a, b)
