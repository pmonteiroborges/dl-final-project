from sys import implementation
from turtle import end_fill
import numpy as np
import tensorflow as tf
from attention import Attention

class PoetryModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(PoetryModel, self).__init__()

        self.vocab_size = vocab_size

        # hyperparameters
        self.encoder_size = 4
        self.decoder_size = 4
        self.embedding_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.attention_input_size = 100 # should be specific numbers to make dimensions lineup
        self.attention_output_size = 100 # same as above

        # embeddings, encoder, decoder, attention, dense layer
        self.encoder_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.decoder_embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)

        self.encoder = tf.keras.layers.GRU(self.encoder_size, return_sequences=True, return_state=True)
        self.attention = Attention(self.attention_input_size, self.attention_output_size)
        self.dense_tanh = tf.keras.layers.Dense(self.encoder_size, activation='tanh')
        self.dense_softmax = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        self.decoder = tf.keras.layers.GRU(self.decoder_size, return_sequences=True)


    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to input sentences
        :param decoder_input: batched ids corresponding to output sentences (should be reversed)

        :return probs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
        """
        # get encoder embeddings and pass into encoder
        encoder_embed = self.encoder_embedding(encoder_input)
        encoder_output, encoder_state = self.encoder(encoder_embed)

        # use encoder states and output to get the context vector
        context_vec = self.attention(encoder_state, encoder_output) # not sure why we put the final state of the encoder here
        concatted = tf.concat([context_vec, encoder_state], axis=-1)

        # should be used as the initial state for the decoder
        dense_output = self.dense_tanh(concatted)

        # get decoder embeddings and pass into decoder
        decoder_embed = self.decoder_embedding(decoder_input)
        decoder_output = self.decoder(decoder_embed, initial_state=dense_output)

        # send through linear layer and softmax to get probabilities
        return self.dense_softmax(decoder_output)

    def accuracy(self, prbs, labels, mask):
        """
		Computes the batch accuracy

		:param prbs: float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
		:param labels: integer tensor, word prediction labels [batch_size x window_size]
		:param mask: tensor that acts as a padding mask [batch_size x window_size]

		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss(self, prbs, labels, mask):
        """
		Calculates the total model cross-entropy loss after one forward pass.
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs: float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
		:param labels: integer tensor, word prediction labels [batch_size x window_size]
		:param mask: tensor that acts as a padding mask [batch_size x window_size]

		:return: the loss of the model as a tensor
		"""
        scce = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs) * mask

        return tf.reduce_sum(scce)
