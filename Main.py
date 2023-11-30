import numpy as np
import Preprocessing as pr
import Dense
np.set_printoptions(threshold=np.inf) # allows whole arrays to be printed (for debugging)
np.random.seed(509)

# Loss functions
def cross_entropy_loss(pred_out: np.ndarray, true_out: np.ndarray, epsilon=1e-15):
    pred_out = np.clip(pred_out, epsilon, 1 - epsilon)
    loss = -np.sum(true_out * np.log(pred_out) + (1 - true_out) * np.log(1 - pred_out))
    return loss


def cross_entropy_derivative(pred_out: np.ndarray, true_out: np.ndarray, epsilon=1e-15):
    pred_out = np.clip(pred_out, epsilon, 1 - epsilon)
    dloss = -(true_out / pred_out) + ((1 - true_out) / (1 - pred_out))
    return dloss


# Training algorithm
def train(first_layer: Dense.Dense, last_layer: Dense.Dense, train_data: list, test_data: list, num_examples=10000, epochs=20, loss_derivative=cross_entropy_derivative):
    #training
    for e in range(epochs):
        num_correct = 0

        for i in range(num_examples):
            example = train_data[i]
            layer_input = example[0]
            true_out = example[1]

            pred_out, oh_pred_out = first_layer.get_output(layer_input)

            if np.array_equal(oh_pred_out, true_out):
                num_correct += 1

            dlda = loss_derivative(pred_out, true_out)
            last_layer.back_propagate(dlda)

        accuracy = num_correct / num_examples
        print('epoch ', e, ': ', accuracy)

    #testing
    num_correct = 0

    for example in test_data:
        layer_input = example[0]
        true_out = example[1]

        pred_out, oh_pred_out = first_layer.get_output(layer_input)

        if np.array_equal(oh_pred_out, true_out):
            num_correct += 1

    accuracy = num_correct / num_examples
    print('test accuracy: ', accuracy)


# <> Process data
# Expects training and testing data in .npy files, I used MNIST.
# For this project I kept the data (4 files) in a local directory called "MNIST", and access it below
train_images, train_labels = np.load(r'MNIST\train_images.npy'), np.load(r'MNIST\train_labels.npy')
train_data = pr.process_data(train_images, train_labels, 10)
test_images, test_labels = np.load(r'MNIST\test_images.npy'), np.load(r'MNIST\test_labels.npy')
test_data = pr.process_data(test_images, test_labels, 10)


# <> Create NN
L1 = Dense.Dense(16, Dense.Dense.relu, 1e-2, 1e-4)
L2 = Dense.Dense(10, Dense.Dense.softmax, 1e-2, 1e-4)
L1.link(None, L2, 784)
L2.link(L1, None)


# <> Train
train(L1, L2, train_data, test_data)