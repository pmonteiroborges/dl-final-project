import numpy as np
import tensorflow as tf
import numpy as np

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_corpus(inputs, labels):
	"""
	DO NOT CHANGE:
	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*". English is padded with "*START*" at the beginning for Teacher Forcing.
	:param french: list of French sentences
	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""

	INPUT_padded_sentences = []
	for line in inputs:
		padded_INPUT = line[:WINDOW_SIZE]
		padded_INPUT = [START_TOKEN] + padded_INPUT + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_INPUT)-1)
		INPUT_padded_sentences.append(padded_INPUT)

	LABEL_padded_sentences = []
	for line in labels:
		padded_LABEL = line[:WINDOW_SIZE]
		padded_LABEL = [START_TOKEN] + padded_LABEL + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_LABEL)-1)
		LABEL_padded_sentences.append(padded_LABEL)

	return INPUT_padded_sentences, LABEL_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE
  Builds vocab from list of sentences
	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE
  Convert sentences to indexed
	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE
  Load text data from file
	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	return text


def get_data(training_file, test_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	:param french_training_file: Path to the French training file.
	:param english_training_file: Path to the English training file.
	:param french_test_file: Path to the French test file.
	:param english_test_file: Path to the English test file.

	:return: Tuple of train containing:
	(2-d list or array with English training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with English test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with French training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with French test sentences in vectorized/id form [num_sentences x 14]),
	English vocab (Dict containg word->index mapping),
	French vocab (Dict containg word->index mapping),
	English padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""

	train_inputs = read_data(training_file)
	train_labels = train_inputs[1:]
	train_inputs = train_inputs[:-1]

	train_inputs, train_labels = pad_corpus(train_inputs, train_labels)

	test_inputs = read_data(test_file)
	test_labels = test_inputs[1:]
	test_inputs = test_inputs[:-1]
	test_inputs, test_labels = pad_corpus(test_inputs, test_labels)

	vocab, padding_index = build_vocab(train_inputs)

	train_inputs = convert_to_id(vocab, train_inputs)
	train_labels = convert_to_id(vocab, train_labels)
	test_labels = convert_to_id(vocab, test_labels)
	test_inputs = convert_to_id(vocab, test_inputs)

	return train_inputs, train_labels, test_inputs, test_labels, vocab, padding_index
