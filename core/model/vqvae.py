import tensorflow as tf
import keras as keras
from keras import layers

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook embeddings
        initializer = keras.initializers.VarianceScaling()
        self.embeddings = self.add_weight(
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=initializer,
            trainable=True,
            name="embeddings"
        )

    def call(self, inputs):
        # Flatten input
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Calculate distances
        distances = (
            tf.reduce_sum(flat_inputs ** 2, axis=1, keepdims=True) -
            2 * tf.matmul(flat_inputs, self.embeddings, transpose_b=True) +
            tf.reduce_sum(self.embeddings ** 2, axis=1)
        )

        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # Quantize
        quantized = tf.matmul(encodings, self.embeddings)
        quantized = tf.reshape(quantized, tf.shape(inputs))

        # Loss
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        self.add_loss(loss)

        # Straight-through estimator
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        return quantized

def build_vqvae(input_shape, embedding_dim, num_embeddings, commitment_cost):
    encoder_inputs = keras.Input(shape=input_shape)  # e.g. (17, 1387, 1)

    # ====== Encoder ======
    x = layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu')(encoder_inputs)  # -> (ceil(17/2), ceil(1387/2))
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu')(x)               # -> (ceil(H/4), ceil(W/4))
    x = layers.Conv2D(embedding_dim, kernel_size=1, strides=1, padding='same')(x)

    # ====== VQ Layer ======
    quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
    quantized = quantizer(x)

    # ====== Decoder ======
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', activation='relu')(quantized)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding='same', activation='relu')(x)

    # Resize về kích thước input ban đầu
    x = layers.Resizing(17, 1387)(x)

    decoder_outputs = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')(x)


    vqvae = keras.Model(encoder_inputs, decoder_outputs, name='VQ-VAE')

    return vqvae