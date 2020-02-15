import numpy as np
from tkinter import *
import keyboard
import time


# NEURAL NETWORK SETTINGS

# topological settings:
amount_of_layers = 5
amount_of_input_neurons = 8 ** 2
amount_of_output_neurons = 9

# activation function settings:
sensitivity = 0.5
magnitude_constant = 0.5
shift = 0.5

# score dependant behaviour:
velocity_sensitivity_relative_to_score = 50 ** -1
general_velocity = 0.5

# learning conditions:
amount_of_neural_networks = 5
amount_of_iterations = 0
maximum_begin_radius_for_weights = 5  # the largest value the weights can have when generated in  the beginning
maximum_begin_radius_for_bias = 5  # the largest value the bias can have when generated in the beginning
learning_rate = 0.5  # a factor of the neural networks step size
ratio_of_reset_networks = 0.5  # the ratio of all of the networks to be reset and the total amount
maximum_reset_radius = 5  # the maximum radius between the reset networks and their models
reset_network_cluster_density_around_best_model = 1  # the larger the value, the larger the focus around better networks

# user input dependant variables (not changeable):
amount_of_neurons_in_layer = 0
amount_of_reset_networks = 0


def set_variables():  # sets the necessary variables in accordance with the user input
    global amount_of_neurons_in_layer, amount_of_reset_networks

    amount_of_neurons_in_layer = np.repeat(0, amount_of_layers)

    for index in range(amount_of_layers):  # the loop creates an array carrying the neuron count of each layer
        amount_of_neurons_in_layer[index] = round((amount_of_input_neurons - amount_of_output_neurons) * \
                                                  np.sin(1.5 * np.pi * (index / (amount_of_layers - 1))) + amount_of_input_neurons)

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

    def x(self):  # determines velocity scalar dependant on the networks score: modifiable function
        velocity_factor = general_velocity
        return velocity_factor

    def propagation(self):  # determines the output values of the network: modifiable function
        for index0 in range(amount_of_layers - 1):
            self.layers[index0 + 1] = (self.weights[index0].dot(self.layers[index0])) + self.bias[index0]
            self.layers[index0 + 1] = self.activation_function(self.layers[index0 + 1])

    def return_coord_of_nth_largest_output_value(self, nth_largest):  # returns the coordinates of the nth largest value in
        # the output layer
        coord_of_nth_largest_value = self.layers[amount_of_layers - 1].argsort()[::-1][:][nth_largest]
        return coord_of_nth_largest_value


class Task:
    """
    This class defines the task given to the networks: All Is Modifiable
    """
    def __init__(self):  # defines certain task properties
        self.amount_of_slices = 15
        self.wall_tile = 10
        self.game_field = 0
        self.player_0_coord = 0
        self.player_1_coord = 0

        self.set_and_reset_variables()

    def set_and_reset_variables(self):
        self.game_field = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, self.wall_tile, self.wall_tile, self.wall_tile, self.wall_tile, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, self.wall_tile, self.wall_tile, self.wall_tile, self.wall_tile, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]])

        self.player_0_coord = np.array([7, 0])
        self.player_1_coord = np.array([0, 0])

    def perform_task(self, contesting_network0, contesting_network1):  # performs the task while simultaneously assigning all the to their
        # corresponding networks
        for index0 in range(amount_of_neural_networks):
            for index1 in range(amount_of_neural_networks):
                self.set_and_reset_variables()
                self.game_field[self.player_0_coord[0]][self.player_0_coord[1]] = 100
                self.game_field[self.player_1_coord[0]][self.player_1_coord[1]] = -100

                for index2 in range(150):
                    contesting_network0[index0].layers[0] = self.game_field.flatten()
                    contesting_network0[index0].propagation()
                    output = contesting_network0[index0].return_coord_of_nth_largest_output_value(0)
                    step = np.array([(output % 3) - 1, int(output / 3) - 1])

                    if 0 <= self.player_0_coord[0] + step[0] <= (self.game_field.shape[0] - 1) and 0 <= \
                    self.player_0_coord[1] + step[1] <= (self.game_field.shape[1] - 1):
                        if self.game_field[self.player_0_coord[0] + step[0]][self.player_0_coord[1] + \
                        step[1]] != self.wall_tile:
                            self.game_field[self.player_0_coord[0]][self.player_0_coord[1]] = 0
                            self.player_0_coord = self.player_0_coord + step
                            self.game_field[self.player_0_coord[0]][self.player_0_coord[1]] = 100

                            for index3 in range(self.amount_of_slices - 1, -1, -1):
                                if magnitude(self.player_1_coord - self.player_0_coord) <= (magnitude(np.array([ \
                                self.game_field.shape[0], self.game_field.shape[1]])) * index3) / \
                                (self.amount_of_slices - 1):
                                    contesting_network0[index0].score = contesting_network0[index0].score + \
                                                                       int(round((self.amount_of_slices - 1) / \
                                                                       (index3 + 0.2 * (self.amount_of_slices - 1))))

                                else:
                                    break

                    contesting_network1[index1].layers[0] = self.game_field.flatten()
                    contesting_network1[index1].propagation()
                    output = contesting_network1[index1].return_coord_of_nth_largest_output_value(0)
                    step = np.array([(output % 3) - 1, int(output / 3) - 1])

                    if 0 <= self.player_0_coord[0] + step[0] <= (self.game_field.shape[0] - 1) and 0 <= \
                            self.player_0_coord[1] + step[1] <= (self.game_field.shape[1] - 1):
                        if self.game_field[self.player_0_coord[0] + step[0]][self.player_0_coord[1] + \
                                                                             step[1]] != self.wall_tile:
                            self.game_field[self.player_0_coord[0]][self.player_0_coord[1]] = 0
                            self.player_0_coord = self.player_0_coord + step
                            self.game_field[self.player_0_coord[0]][self.player_0_coord[1]] = -100

                            for index3 in range(self.amount_of_slices):
                                if magnitude(self.player_1_coord - self.player_0_coord) >= (magnitude(np.array([ \
                                        self.game_field.shape[0], self.game_field.shape[1]])) * index3) / \
                                        (self.amount_of_slices - 1):
                                    contesting_network1[index1].score = contesting_network1[index1].score + \
                                                                        int(round((self.amount_of_slices - 1) / \
                                                                                  ((self.amount_of_slices - index3) + \
                                                                                   0.2 * (self.amount_of_slices - 1))))

                                else:
                                    break


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
            score_information = np.insert(score_information, index0 - amount_of_reset_networks, \
                                          reset_network_cluster_density_around_best_model * \
                                          ((self.neural_network[score_catalog[index0]].score ** 3) / 2500000))

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

    def let_score_equal_zero(self):
        for index0 in range(amount_of_neural_networks):
            self.neural_network[index0].score = 0

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
                                                  self.neural_network[index0].x() * change_in_weights
            self.neural_network[index0].bias = self.neural_network[index0].bias + learning_rate * \
                                               self.neural_network[index0].x() * change_in_bias


training_over = False


def break_loop():
    global training_over

    training_over = True
    return ''


def main():  # the center of coordination
    global training_over, amount_of_neural_networks

    print('Amount of contesting neural networks: ')
    amount_of_neural_networks = int(input())
    print('')

    set_variables()
    neural_network_coordinator0 = NeuralNetworkCoordinator()
    neural_network_coordinator1 = NeuralNetworkCoordinator()
    the_task = Task()

    iteration = 0

    while True:
        training_over = False

        while True:  # trains the networks
            neural_network_coordinator0.let_score_equal_zero()
            neural_network_coordinator1.let_score_equal_zero()
            the_task.perform_task(neural_network_coordinator0.neural_network, neural_network_coordinator1.neural_network)
            neural_network_coordinator0.learn()
            neural_network_coordinator1.learn()

            print("Catcher's best score: ", neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks - 1]].score)
            print("Runner's best score: ", neural_network_coordinator1.neural_network[neural_network_coordinator1.score_order[amount_of_neural_networks - 1]].score)

            keyboard.on_press_key('o', lambda _:print(break_loop()))

            if training_over:
                break

            print(iteration)
            iteration = iteration + 1

        print("weights0:")  # prints the best networks weight values
        print(neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks \
                                                                                               - 1]].weights)
        print("bias0:")  # prints the best networks bias values
        print(neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks \
                                                                                               - 1]].bias)

        game_field_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, the_task.wall_tile, \
                                         the_task.wall_tile, \
                                         the_task.wall_tile, \
                                         the_task.wall_tile, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, the_task.wall_tile, \
                                         the_task.wall_tile, \
                                         the_task.wall_tile, \
                                         the_task.wall_tile, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0]])

        learn = False

        while True:
            print('')
            print('Would you like to try to catch the ai? (y/n): ')

            answer = input()

            if answer == 'n':
                ai_0 = True

                human_x_coord_0 = 7 * 64
                human_y_coord_0 = 0
                human_x_coord_1 = 8 * 64
                human_y_coord_1 = 64
                human_board_coord = np.array([7, 0])

                ai_x_coord_0 = 0
                ai_y_coord_0 = 0
                ai_x_coord_1 = 64
                ai_y_coord_1 = 64
                ai_board_coord = np.array([0, 0])

            if answer == 'y':
                ai_0 = False

                human_x_coord_0 = 0
                human_y_coord_0 = 0
                human_x_coord_1 = 64
                human_y_coord_1 = 64
                human_board_coord = np.array([0, 0])

                ai_x_coord_0 = 7 * 64
                ai_y_coord_0 = 0
                ai_x_coord_1 = 8 * 64
                ai_y_coord_1 = 64
                ai_board_coord = np.array([7, 0])

            tkinter = Tk()
            game_field = Canvas(tkinter, width=8 ** 3, height=8 ** 3)
            tkinter.title("Catch Game")
            game_field.pack()

            outer_boundary = game_field.create_rectangle(0, 0, 8 * 64, 8 * 64, fill="white")

            human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0, human_x_coord_1, human_y_coord_1,
                                                          fill="dark blue")
            ai_character = game_field.create_rectangle(ai_x_coord_0, ai_y_coord_0, ai_x_coord_1, ai_y_coord_1, fill="dark red")
            wall0 = game_field.create_rectangle(2 * 64, 64, 3 * 64, 5 * 64, fill="black")
            wall1 = game_field.create_rectangle(5 * 64, 2 * 64, 6 * 64, 6 * 64, fill="black")
            tkinter.update()


            while True:
                if ai_0:
                    if keyboard.is_pressed('l'):
                        learn = True
                        break

                    if keyboard.is_pressed('o'):
                        game_field_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0]])

                        game_field.delete(human_character)
                        game_field.delete(ai_character)

                        human_x_coord_0 = 0
                        human_y_coord_0 = 0
                        human_x_coord_1 = 64
                        human_y_coord_1 = 64
                        human_board_coord = np.array([0, 0])

                        ai_x_coord_0 = 7 * 64
                        ai_y_coord_0 = 0
                        ai_x_coord_1 = 8 * 64
                        ai_y_coord_1 = 64
                        ai_board_coord = np.array([7, 0])

                        human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0, human_x_coord_1,
                                                                      human_y_coord_1,
                                                                      fill="dark blue")
                        ai_character = game_field.create_rectangle(ai_x_coord_0, ai_y_coord_0, ai_x_coord_1, ai_y_coord_1,
                                                                   fill="dark red")

                        game_field_array[human_board_coord[0]][human_board_coord[1]] = 100
                        game_field_array[ai_board_coord[0]][ai_board_coord[1]] = -100

                        ai_0 = False

                        print("")
                        print("Try to catch the ai.")
                        print("")

                        tkinter.update()

                    if keyboard.is_pressed('a'):
                        if 0 <= (human_board_coord[0] - 1) <= 7:
                            if game_field_array[human_board_coord[0] - 1][human_board_coord[1]] != \
                            the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[0] = human_board_coord[0] - 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_x_coord_0 = human_x_coord_0 - 64
                                human_x_coord_1 = human_x_coord_1 - 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0, human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('d'):
                        if 0 <= (human_board_coord[0] + 1) <= 7:
                            if game_field_array[human_board_coord[0] + 1][human_board_coord[1]] != \
                            the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[0] = human_board_coord[0] + 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_x_coord_0 = human_x_coord_0 + 64
                                human_x_coord_1 = human_x_coord_1 + 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('w'):
                        if 0 <= (human_board_coord[1] - 1) <= 7:
                            if game_field_array[human_board_coord[0]][human_board_coord[1] - 1] != \
                            the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[1] = human_board_coord[1] - 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_y_coord_0 = human_y_coord_0 - 64
                                human_y_coord_1 = human_y_coord_1 - 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('s'):
                        if 0 <= (human_board_coord[1] + 1) <= 7:
                            if game_field_array[human_board_coord[0]][human_board_coord[1] + 1] != \
                            the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[1] = human_board_coord[1] + 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_y_coord_0 = human_y_coord_0 + 64
                                human_y_coord_1 = human_y_coord_1 + 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks - 1]].layers[0] = game_field_array.flatten()
                    neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks - 1]].propagation()
                    output = neural_network_coordinator0.neural_network[neural_network_coordinator0.score_order[amount_of_neural_networks - 1]].return_coord_of_nth_largest_output_value(0)
                    step = np.array([(output % 3) - 1, int(output / 3) - 1])

                    print(step)

                    if 0 <= ai_board_coord[0] + step[0] <= 7 and 0 <= ai_board_coord[1] + step[1] <= 7:
                        if game_field_array[ai_board_coord[0] + step[0]][ai_board_coord[1] + step[1]] != \
                        the_task.wall_tile:
                            game_field_array[ai_board_coord[0]][ai_board_coord[1]] = 0
                            ai_board_coord = ai_board_coord + step
                            game_field_array[ai_board_coord[0]][ai_board_coord[1]] = 100

                            ai_x_coord_0 = ai_x_coord_0 + 64 * step[0]
                            ai_x_coord_1 = ai_x_coord_1 + 64 * step[0]
                            ai_y_coord_0 = ai_y_coord_0 + 64 * step[1]
                            ai_y_coord_1 = ai_y_coord_1 + 64 * step[1]
                            game_field.delete(ai_character)
                            ai_character = game_field.create_rectangle(ai_x_coord_0, ai_y_coord_0, ai_x_coord_1, ai_y_coord_1, fill="dark red")

                    tkinter.update()

                    time.sleep(0.25)

                if ai_0 is False:
                    if keyboard.is_pressed('l'):
                        learn = True
                        break

                    if keyboard.is_pressed('o'):
                        game_field_array = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, \
                                                      the_task.wall_tile, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0]])

                        game_field.delete(human_character)
                        game_field.delete(ai_character)

                        human_x_coord_0 = 7 * 64
                        human_y_coord_0 = 0
                        human_x_coord_1 = 8 * 64
                        human_y_coord_1 = 64
                        human_board_coord = np.array([7, 0])

                        ai_x_coord_0 = 0
                        ai_y_coord_0 = 0
                        ai_x_coord_1 = 64
                        ai_y_coord_1 = 64
                        ai_board_coord = np.array([0, 0])

                        human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0, human_x_coord_1,
                                                                      human_y_coord_1,
                                                                      fill="dark blue")
                        ai_character = game_field.create_rectangle(ai_x_coord_0, ai_y_coord_0, ai_x_coord_1,
                                                                   ai_y_coord_1,
                                                                   fill="dark red")

                        game_field_array[human_board_coord[0]][human_board_coord[1]] = -100
                        game_field_array[ai_board_coord[0]][ai_board_coord[1]] = 100

                        ai_0 = True

                        print("")
                        print("The ai will try to catch you.")
                        print("")

                        tkinter.update()

                    if keyboard.is_pressed('a'):
                        if 0 <= (human_board_coord[0] - 1) <= 7:
                            if game_field_array[human_board_coord[0] - 1][human_board_coord[1]] != \
                                    the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[0] = human_board_coord[0] - 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_x_coord_0 = human_x_coord_0 - 64
                                human_x_coord_1 = human_x_coord_1 - 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('d'):
                        if 0 <= (human_board_coord[0] + 1) <= 7:
                            if game_field_array[human_board_coord[0] + 1][human_board_coord[1]] != \
                                    the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[0] = human_board_coord[0] + 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_x_coord_0 = human_x_coord_0 + 64
                                human_x_coord_1 = human_x_coord_1 + 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('w'):
                        if 0 <= (human_board_coord[1] - 1) <= 7:
                            if game_field_array[human_board_coord[0]][human_board_coord[1] - 1] != \
                                    the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[1] = human_board_coord[1] - 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_y_coord_0 = human_y_coord_0 - 64
                                human_y_coord_1 = human_y_coord_1 - 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    if keyboard.is_pressed('s'):
                        if 0 <= (human_board_coord[1] + 1) <= 7:
                            if game_field_array[human_board_coord[0]][human_board_coord[1] + 1] != \
                                    the_task.wall_tile:
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = 0
                                human_board_coord[1] = human_board_coord[1] + 1
                                game_field_array[human_board_coord[0]][human_board_coord[1]] = -100

                                human_y_coord_0 = human_y_coord_0 + 64
                                human_y_coord_1 = human_y_coord_1 + 64
                                game_field.delete(human_character)
                                human_character = game_field.create_rectangle(human_x_coord_0, human_y_coord_0,
                                                                              human_x_coord_1,
                                                                              human_y_coord_1, fill="dark blue")
                                tkinter.update()

                    neural_network_coordinator1.neural_network[
                        neural_network_coordinator1.score_order[amount_of_neural_networks - 1]].layers[
                        0] = game_field_array.flatten()
                    neural_network_coordinator1.neural_network[
                        neural_network_coordinator1.score_order[amount_of_neural_networks - 1]].propagation()
                    output = neural_network_coordinator1.neural_network[neural_network_coordinator1.score_order[
                        amount_of_neural_networks - 1]].return_coord_of_nth_largest_output_value(0)
                    step = np.array([(output % 3) - 1, int(output / 3) - 1])

                    print(step)

                    if 0 <= ai_board_coord[0] + step[0] <= 7 and 0 <= ai_board_coord[1] + step[1] <= 7:
                        if game_field_array[ai_board_coord[0] + step[0]][ai_board_coord[1] + step[1]] != \
                                the_task.wall_tile:
                            game_field_array[ai_board_coord[0]][ai_board_coord[1]] = 0
                            ai_board_coord = ai_board_coord + step
                            game_field_array[ai_board_coord[0]][ai_board_coord[1]] = 100

                            ai_x_coord_0 = ai_x_coord_0 + 64 * step[0]
                            ai_x_coord_1 = ai_x_coord_1 + 64 * step[0]
                            ai_y_coord_0 = ai_y_coord_0 + 64 * step[1]
                            ai_y_coord_1 = ai_y_coord_1 + 64 * step[1]
                            game_field.delete(ai_character)
                            ai_character = game_field.create_rectangle(ai_x_coord_0, ai_y_coord_0, ai_x_coord_1,
                                                                       ai_y_coord_1, fill="dark red")

                    tkinter.update()

                    time.sleep(0.25)

            if learn is True:
                game_field.delete(wall0)
                game_field.delete(wall1)
                game_field.delete(human_character)
                game_field.delete(ai_character)
                game_field.delete(outer_boundary)
                tkinter.destroy()

                print('')
                break


if __name__ == "__main__":
    main()
