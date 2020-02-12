import numpy as np


# NEURAL NETWORK SETTINGS

# topological settings:
amount_of_layers = 3
amount_of_input_neurons = 0
amount_of_output_neurons = 0

# activation function settings:
sensitivity = 0.5
magnitude_constant = 0.5
shift = 0.5

# score dependant behaviour:
velocity_sensitivity_relative_to_score = 50 ** -1
general_velocity = 0.5

# learning conditions:
amount_of_neural_networks = 50
amount_of_iterations = 150
maximum_begin_radius_for_weights = 5  # the largest value the weights can have when generated in  the beginning
maximum_begin_radius_for_bias = 5  # the largest value the bias can have when generated in the beginning
learning_rate = 0.5  # a factor of the neural networks step size
ratio_of_reset_networks = 2/5  # the ratio of all of the networks to be reset and the total amount
maximum_reset_radius = 5  # the maximum radius between the reset networks and their models
reset_network_cluster_density_around_best_model = 2

# user input dependant variables (not changeable):
amount_of_neurons_in_layer = 0
amount_of_reset_networks = 0


def set_variables():  # sets the necessary variables in accordance with the user input
    global amount_of_neurons_in_layer, amount_of_reset_networks

    amount_of_neurons_in_layer = np.repeat(0, amount_of_layers)

    for index in range(amount_of_layers):  # the loop creates an array carrying the neuron count of each layer
        amount_of_neurons_in_layer[index] = round(-index/2 + 2)

    amount_of_reset_networks = int(ratio_of_reset_networks * amount_of_neural_networks)


def magnitude(arbitrary_array):  # this method computes the magnitude of any array given as an input
    magnitude_of_arbitrary_array = np.array([], dtype='float')

    for index0 in range(arbitrary_array.shape[0]):
        magnitude_of_arbitrary_array = np.insert(magnitude_of_arbitrary_array, index0, np.sum(arbitrary_array[index0] \
                                                                                              ** 2))

    magnitude_of_arbitrary_array = np.sum(magnitude_of_arbitrary_array) ** 0.5

    if magnitude_of_arbitrary_array == 0:
        magnitude_of_arbitrary_array = 1

    return magnitude_of_arbitrary_array


class NeuralNetwork:
    """
    This class creates the neural network described by user above
    """
    def __init__(self):  # setting the network up
        self.layers = []
        self.weights = []
        self.bias = []
        self.score = 0

        for index0 in range(amount_of_layers):  # assigns the "layers" array all necessary neurons for each layer
            self.layers.append(np.repeat(0.0, amount_of_neurons_in_layer[index0]))

        self.layers = np.asarray(self.layers)

        for index0 in range(1, amount_of_layers):  # assigns the "weights" and "bias" arrays all their dimensions
            self.weights.append(2 * maximum_begin_radius_for_weights * (np.random.rand( \
                amount_of_neurons_in_layer[index0], amount_of_neurons_in_layer[index0 - 1]) - 0.5))
            self.bias.append(2 * maximum_begin_radius_for_bias * (np.random.rand(amount_of_neurons_in_layer[index0]) \
                                                                  - 0.5))

        self.weights = np.asarray(self.weights)
        self.bias = np.asarray(self.bias)

    @staticmethod
    def activation_function(weighted_sum):  # the activation curve: modifiable function
        activation = magnitude_constant * np.tanh(sensitivity * weighted_sum) + shift
        return activation

    def eta(self):  # determines velocity scalar dependant on the networks score: modifiable function
        velocity_factor = 2 * general_velocity * (np.cosh(velocity_sensitivity_relative_to_score * self.score) ** -1)
        return velocity_factor

    def propagation(self):  # determines the output values of the network: modifiable function
        for index0 in range(amount_of_layers - 1):
            self.layers[index0 + 1] = (self.weights[index0].dot(self.layers[index0])) + self.bias[index0]
            self.layers[index0 + 1] = self.activation_function(self.layers[index0 + 1])

    def return_coord_of_largest_output_neuron(self, nth_largest):  # returns the coordinates of the nth largest value in
        # the output layer
        coord_of_nth_largest_value = self.layers[amount_of_layers - 1].argsort()[::-1][:][nth_largest]
        return coord_of_nth_largest_value


class Task:
    """
    This class defines the task given to the networks: All Is Modifiable
    """
    def __init__(self):  # defines certain task properties
        self.amount_of_slices = 50
        self.input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.output = np.array([0, 1, 1, 0])
        self.highest_possible_score = self.amount_of_slices * self.output.size

    def perform_task(self, contesting_network):  # performs the task while simultaneously assigning all the to their
        # corresponding networks
        for index0 in range(amount_of_neural_networks):
            score_vector = np.array([])

            for index1 in range(self.output.size):
                contesting_network[index0].layers[0] = self.input[index1]
                contesting_network[index0].propagation()

                score_vector = np.insert(score_vector, index1, contesting_network[index0].layers[amount_of_layers - 1][0])

            for index1 in range(self.amount_of_slices):
                if magnitude(score_vector - self.output) <= ((index1 * (2 ** 0.5)) / (self.amount_of_slices - 1)):
                    contesting_network[index0].score = contesting_network[index0].score + 1


class NeuralNetworkCoordinator:
    """
    This class is responsible for coordinating the learning process of all neural networks
    """
    def __init__(self):  # creates the task and all neural networks
        self.task = Task()
        self.neural_network = np.array([NeuralNetwork()])

        for index0 in range(1, amount_of_neural_networks):
            self.neural_network = np.insert(self.neural_network, index0, NeuralNetwork())

        self.score_order = np.array([])

    def reset_network_cluster(self, score_catalog):  # creates array containing the reset networks cluster density
        # profile: modifiable function
        reset_network_cluster_density_profile = np.array([score_catalog[amount_of_neural_networks - 1]])
        score_information = np.array([])

        for index0 in range(amount_of_reset_networks, amount_of_neural_networks):
            score_information = np.insert(score_information, index0 - amount_of_reset_networks, np.e ** \
                                          -(reset_network_cluster_density_around_best_model * \
                                            (self.task.highest_possible_score - \
                                             self.neural_network[score_catalog[index0]].score)))

        score_information_sum = score_information.sum()

        if score_information_sum == 0:
            score_information_sum = 1

        for index0 in range(amount_of_reset_networks, amount_of_neural_networks):
            reset_network_cluster_density_profile = np.insert(reset_network_cluster_density_profile, 0, np.repeat( \
                index0, int((score_information[index0 - amount_of_reset_networks] / score_information_sum) * \
                            amount_of_reset_networks)))

        return reset_network_cluster_density_profile[np.random.randint(reset_network_cluster_density_profile.size)]

    def evaluate_score(self):  # evaluates the score of each network
        for index0 in range(amount_of_neural_networks):
            self.neural_network[index0].score = 0

        self.task.perform_task(self.neural_network)

    def relevance(self, referred_network):  # determines the relevance of a network by comparing the scores
        sum_of_all_scores = 0

        for index0 in range(amount_of_neural_networks):
            sum_of_all_scores = sum_of_all_scores + self.neural_network[index0].score

        if sum_of_all_scores == 0:
            sum_of_all_scores = 1

        return self.neural_network[referred_network].score/sum_of_all_scores

    def learn(self):  # performs a learning step
        self.score_order = np.array([])

        for index0 in range(amount_of_neural_networks):  # enters all values into the array: self.score_order
            self.score_order = np.insert(self.score_order, index0, self.neural_network[index0].score)

        self.score_order = np.argsort(self.score_order)  # orders all values in the array: self.score_order

        for index0 in range(amount_of_reset_networks):
            current_model_network = self.neural_network[self.reset_network_cluster(self.score_order)]
            random_add_on_weights = []
            random_add_on_bias = []

            for index1 in range(1, amount_of_layers):  # resets all of the worst networks in accordance with the cluster
                # density function
                random_add_on_weights.append(2 * maximum_begin_radius_for_weights * (np.random.rand( \
                    amount_of_neurons_in_layer[index1], amount_of_neurons_in_layer[index1 - 1]) - 0.5))
                random_add_on_bias.append(2 * maximum_begin_radius_for_bias * (np.random.rand( \
                    amount_of_neurons_in_layer[index1]) - 0.5))

            random_add_on_weights = np.asarray(random_add_on_weights)
            random_add_on_bias = np.asarray(random_add_on_bias)

            self.neural_network[self.score_order[index0]].weights = current_model_network.weights + \
                                                                       random_add_on_weights
            self.neural_network[self.score_order[index0]].bias = current_model_network.bias + random_add_on_bias

        for index0 in range(amount_of_reset_networks, amount_of_neural_networks - 1):  # changes the weights and bias of
            # all but the best network
            change_in_weights = 0
            change_in_bias = 0

            for index1 in range(amount_of_reset_networks, amount_of_neural_networks - 1):  # computes the change in the
                # weights and bias
                change_in_weights = change_in_weights + self.relevance(index1) * ((self.neural_network[index1 \
                                                                                       ].weights - \
                                                                                   self.neural_network[index0 \
                                                                                       ].weights) / magnitude( \
                    self.neural_network[index1].weights - self.neural_network[index0].weights))
                change_in_bias = change_in_bias + self.relevance(index1) * ((self.neural_network[index1].bias - \
                                                                             self.neural_network[index0].bias) / \
                                                                            magnitude(self.neural_network[index1 \
                                                                                          ].bias - \
                                                                                      self.neural_network[index0].bias))

            self.neural_network[index0].weights = self.neural_network[index0].weights + learning_rate * \
                                                  self.neural_network[index0].eta() * change_in_weights
            self.neural_network[index0].bias = self.neural_network[index0].bias + learning_rate * \
                                               self.neural_network[index0].eta() * change_in_bias


def main():  # the center of coordination
    set_variables()
    neural_network_coordinator = NeuralNetworkCoordinator()

    for iteration in range(amount_of_iterations):  # trains the networks
        neural_network_coordinator.evaluate_score()
        neural_network_coordinator.learn()

        print(iteration)

    print("weights:")  # prints the best networks weight values
    print(neural_network_coordinator.neural_network[neural_network_coordinator.score_order[amount_of_neural_networks \
                                                                                           - 1]].weights)
    print("bias:")  # prints the best networks bias values
    print(neural_network_coordinator.neural_network[neural_network_coordinator.score_order[amount_of_neural_networks \
                                                                                           - 1]].bias)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for index0 in range(4):
        print('')

        neural_network_coordinator.neural_network[neural_network_coordinator.score_order[amount_of_neural_networks - 1 \
            ]].layers[0] = inputs[index0]
        neural_network_coordinator.neural_network[neural_network_coordinator.score_order[amount_of_neural_networks - 1 \
            ]].propagation()

        print(inputs[index0], ':')
        print(round(neural_network_coordinator.neural_network[neural_network_coordinator.score_order[ \
            amount_of_neural_networks - 1]].layers[amount_of_layers - 1][0]))


if __name__ == "__main__":
    main()
