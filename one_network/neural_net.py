import numpy as np
import activation_functions as af


class SingleLayer:
    def __init__(self, input_size, output_size, learning_rate=0.001, activation=None):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        # randomly initialize weights for this layer
        self.weights = np.random.uniform(
            low=-0.5, high=0.5, size=(input_size, output_size))

    # perform forward propagation for this single layer and output results for next layer to use
    def forward_propagation_single(self, inputs, memorize=True):

        biased_input = np.append(inputs, 1)
        raw_output = np.dot(biased_input, self.weights)
        activated_output = raw_output
        if self.activation:
            activated_output = self.activation(raw_output)
        if memorize:
            self.backprop_input = biased_input
            self.backprop_output = np.copy(raw_output)
        return activated_output

    def backward_propagation_single(self, gradient):
        adjustment = gradient
        
        if self.activation != None:
            adjustment = np.multiply(af.relu_deriv(self.backprop_output), gradient)

        loss_deriv_to_weights = np.dot(np.transpose(np.reshape(self.backprop_input, (1, len(
            self.backprop_input)))), np.reshape(adjustment, (1, len(adjustment))))
        error = np.dot(adjustment, np.transpose(self.weights))[:-1]

        self.update_weights(loss_deriv_to_weights)
        return error

    def update_weights(self, change):
        self.weights = self.weights - self.learning_rate * change
