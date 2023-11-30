import numpy as np

class Dense:
    def __init__(self, output_size: int, activation_function, learning_rate: float, regularizer: float):
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = Dense.__get_activation_derivative(self.activation_function)
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.linked = False


    # fourth order central difference approximation
    def derivative(f, x: np.ndarray, h=1e-6):
        return (-f(x + (2 * h)) + (8 * f(x + h)) - (8 * f(x - h)) + f(x - (2 * h))) / (12 * h)


    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return Dense.sigmoid(x) * (1 - Dense.sigmoid(x))


    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)


    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)


    def softmax(x: np.ndarray) -> np.ndarray:
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        activations = e_x / np.sum(e_x)
        return activations


    def softmax_derivative(x: np.ndarray) -> np.ndarray:
        return Dense.softmax(x) * (1 - Dense.softmax(x))


    def __get_activation_derivative(f):
        match f:
            case Dense.relu:
                return Dense.relu_derivative
            case Dense.sigmoid:
                return Dense.sigmoid_derivative
            case Dense.softmax:
                return Dense.softmax_derivative
            case _:
                return lambda x: Dense.derivative(f, x)


    def link(self, prev_layer, next_layer, expected_input_size=None):
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        if self.prev_layer is None:
            if expected_input_size is None:
                raise Exception('expected input size required if previous layer is None')
            self.is_first = True
            self.input_size = expected_input_size
        else:
            self.is_first = False
            self.input_size = self.prev_layer.output_size

        if self.next_layer is None:
            self.is_last = True
        else:
            self.is_last = False

        self.linked = True
        self.init_params()


    def init_params(self):
        if self.linked is not True:
            raise Exception('Layer must be linked before initializing parameters')
        
        sigma = np.sqrt(2 / self.input_size) # HE initialization
        self.weights = np.random.normal(0, sigma, (self.output_size, self.input_size))
        self.bias = np.zeros(self.output_size)


    def one_hot(pred_out):
        oh = np.zeros(pred_out.size, dtype=np.uint)
        oh[np.argmax(pred_out)] = 1
        return oh


    def forward(self, layer_input: np.ndarray) -> np.ndarray:
        if self.linked is not True:
            raise Exception('Layer must be linked before forward pass')
        
        self.layer_input = layer_input
        self.z_values = np.matmul(self.weights, self.layer_input) + self.bias
        self.activations = self.activation_function(self.z_values)
        return self.activations
    

    def get_output(self, layer_input: np.ndarray) -> list:
        layer_output = self.forward(layer_input)

        if self.is_last:
            return layer_output, Dense.one_hot(layer_output)
        else:
            return self.next_layer.get_output(layer_output)
        

    def backward(self, dlda: np.ndarray) -> np.ndarray:
        dldz = np.multiply(dlda, self.activation_derivative(self.z_values)) # this is also dldb

        dldw = np.outer(dldz, self.layer_input)
        dldx = np.matmul(self.weights.T, dldz)

        self.weights -= self.learning_rate * dldw + self.regularizer * self.weights # regularize proportionally to L2 norm of weights
        self.bias -= self.learning_rate * dldz
        return dldx


    def back_propagate(self, dlda: np.ndarray) -> np.ndarray:
        dldx = self.backward(dlda)

        if self.is_first:
            return dldx
        else:
            return self.prev_layer.back_propagate(dldx)