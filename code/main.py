import numpy as np
import tensorflow as tf

from model import PoetryModel
from preprocess import get_data


def train(model, train_inputs, train_outputs, padding_index):
    """
    Runs through one epoch - all training examples.

    :param model: the initialized model to use for forward and backward pass
    :param train_inputs: french train data (all data for training) of shape (num_sentences, wimdow_size)
    :param train_outputs: english train data (all data for training) of shape (num_sentences, window_size)
    :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
    :return: None
    """

    num_examples = train_inputs.shape[0]
    num_batches = num_examples - (num_examples % model.batch_size)

    for i in range(0, num_batches, model.batch_size):

        end = i + model.batch_size
        inputs = train_inputs[i:end]
        outputs = train_outputs[i:end, :-1]
        labels = train_outputs[i:end, 1:]

        with tf.GradientTape() as tape:
            logits = model.call(inputs, outputs)
            mask = np.where(padding_index == labels, 0, 1)
            loss = model.loss(logits, labels, mask)

        if i % 10000 == 0:
            print(f"Loss @ {i}th batch: {loss}")

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_outputs, padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initialized model to use for forward and backward pass
    :param test_inputs: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_outputs: english test data (all data for testing) of shape (num_sentences, 15)
    :returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
    e.g. (my_perplexity, my_accuracy)
    """

    # Note: Follow the same procedure as in train() to construct batches of data!
    num_examples = test_inputs.shape[0]
    num_batches = num_examples - (num_examples % model.batch_size)

    total_loss = 0
    total_num_words = 0
    correct_words = 0

    for i in range(0, num_batches, model.batch_size):
        end = i + model.batch_size
        inputs = test_inputs[i:end]
        outputs = test_outputs[i:end, :-1]
        labels = test_outputs[i:end, 1:]

        probabilities = model.call(inputs, outputs)
        mask = np.where(padding_index == labels, 0, 1)

        loss = model.loss(probabilities, labels, mask)
        total_loss += loss

        num_words = np.sum(mask)
        total_num_words += num_words
        accuracy = model.accuracy(probabilities, labels, mask)
        correct_words += accuracy * num_words

    perplexity = tf.math.exp(total_loss / total_num_words)
    accuracy = correct_words / total_num_words

    print(f"train perplexity: {perplexity}")
    print(f"train accuracy: {accuracy}")

    return perplexity, accuracy

def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))

def main():
    data_path = "../data/"
    train_file = "/Users/palomasalseda/Desktop/env/hw1-mnist-paloomers/dl-final-project/data/els.txt" # data_path + "els.txt" 
    test_file = "/Users/palomasalseda/Desktop/env/hw1-mnist-paloomers/dl-final-project/data/elt.txt" #data_path + "elt.txt"

    print("read data")
    train_inputs, train_labels, test_inputs, test_labels, vocab, padding_index = get_data(train_file, test_file)
    print("data recieved and processed")

    model = PoetryModel(len(vocab))

    print("start training")
    train(model, train_inputs, train_labels, padding_index)
    print("model trained")

    print("start testing")
    perplexity, accuracy = test(model, test_inputs, test_labels, padding_index)
    print("final perplexity:", perplexity, "final accuracy:", accuracy)

    #test 1: 
    generate_sentence("Trap", 20, vocab, model)

    #test 2:
    generate_sentence("Sleeping", 15, vocab, model)

    #test 3:
    generate_sentence("Love", 20, vocab, model)

    #test 3:
    generate_sentence("The", 1, vocab, model)


if __name__ == '__main__':
    main()
