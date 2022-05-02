from base64 import encode
import numpy as np
import tensorflow as tf
from attention import Attention
import transformer_funcs as transformer

class Model(tf.keras.Model):
	def __init__(self, vocab_size):

		super(Model, self).__init__()

		self.vocab_size = vocab_size


		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size, optimizer/learning rate
		self.batch_size = 100 #CHANGE
		self.embedding_size = 100
		self.learning_rate = 0.001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.decode_size = 150

		# Define embedding layers:

		self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)

		# Create positional encoder layers
		
		self.position = transformer.Position_Encoding_Layer(self.vocab_size, self.embedding_size)
        #tf.keras.layers.GRU(self.decode_size, return_sequences=True, return_state=True)
        
        self.attention = Attention(self.decode_size)
		
		# Define encoder and decoder layers:

		self.decoder = transformer.Transformer_Block(self.embedding_size, True)
		self.encoder = transformer.Transformer_Block(self.embedding_size, False) #decode size?

		# Define dense layer(s)
        self.dense = tf.keras.layers.Dense(self.decode_size, activation='tanh')
		self.dense2 = tf.keras.layers.Dense(self.decode_size, activation = 'softmax')

	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to input sentences
		:param decoder_input: batched ids corresponding to output sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x vocab_size]
		"""

        embeddings = tf.nn.embedding_lookup(self.embedding, encoder_input)
        eout, efinal_state = self.encoder(embeddings, initial_state=None)

        context = self.attention(efinal_state, eout)
        context_hidden_concat = tf.concat([context, efinal_state], axis=-1)
        hidden_with_attention = self.dense(context_hidden_concat)

        dembeddings = tf.nn.embedding_lookup(self.embeddings, decoder_input)
        dout, dfinal_state = self.decoder(dembeddings,initial_state=hidden_with_attention)

        layer2 = self.dense2(dout)

        return tf.nn.softmax(layer2)

		# #1) Add the positional embeddings to French sentence embeddings
		# french_emb = self.fre(encoder_input)
		# french_pos = self.french_position(french_emb)
		# #2) Pass the French sentence embeddings to the encoder
		# layer1 = self.french_encoder(french_pos)
		# #3) Add positional embeddings to the English sentence embeddings
		# english_emb = self.eng(decoder_input)
		# english_pos = self.english_position(english_emb)
		# #4) Pass the English embeddings and output of your encoder, to the decoder
		# layer2 = self.english_decoder(english_pos, layer1)
		# #5) Apply dense layer(s) to the decoder out to generate probabilities
		# probabilities = self.dense(layer2)

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE
		Computes the batch accuracy

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		masked_loss = tf.boolean_mask(loss, mask) 
		return tf.reduce_sum(masked_loss)
