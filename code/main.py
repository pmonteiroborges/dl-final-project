import numpy as np
import tensorflow as tf

from model import Model
from preprocess import get_data, pad_corpus, convert_to_id


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
        outputs = train_outputs[i:end, 0:model.window_size]
        labels = train_outputs[i:end, 1:model.window_size+1]

        with tf.GradientTape() as tape:
            logits = model.call(inputs, outputs)
            mask = np.where(padding_index == labels, 0, 1)
            loss = model.loss_function(logits, labels, mask)

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
        outputs = test_outputs[i:end, 0:model.window_size]
        labels = test_outputs[i:end, 1:model.window_size+1]

        probabilities = model.call(test_inputs, test_outputs)
        mask = np.where(padding_index == labels, 0, 1)

        loss = model.loss_function(probabilities, labels, mask)
        total_loss += loss

        num_words = np.sum(mask)
        total_num_words += num_words
        accuracy = model.accuracy_function(probabilities, labels, mask)
        correct_words += accuracy * num_words

    perplexity = tf.math.exp(total_loss / total_num_words)
    accuracy = correct_words / total_num_words

    print(f"perplexity: {perplexity}")
    print(f"accuracy: {accuracy}")

    return perplexity, accuracy


def main():
    data_path = "../../data/"
    train_file = data_path + "els.txt"
    test_file = data_path + "elt.txt"
    train_inputs, train_labels, test_inputs, test_labels, vocab, padding_index = get_data(train_file, test_file)
    model = Model(len(vocab_eng))
    train(model, train_inputs, train_labels, padding_index)
    perplexity, accuracy = test(model, test_inputs, test_labels, padding_index)
    print("perplexity:", perplexity, "accuracy:", accuracy)


if __name__ == '__main__':
    main()
