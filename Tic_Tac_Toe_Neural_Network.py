import numpy as np


radius_of_random_generation = 5
learning_rate = 0.1
score_adaptivity = 20 ** -1  # the difference between the networks rate of change when its score is high and when its
# low
number_of_iterations = 100
number_of_networks = 25
ratio_of_reset_networks = 2/5  # the ratio between the neural networks that are reset every round and the that aren't
flip_turn = False  # whether the network starts when you play it


class NeuralNetwork:
    """
    This class defines the properties the neural networks will have.
    """
    def __init__(self):
        self.layers = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float64')
        self.bias = 2*(np.random.rand(3, 9) - 0.5)
        self.weights = 2*(np.random.rand(3, 9, 9) - 0.5)
        self.score = 0

    def sigmoid_activation_function(self, input):
        output = ((np.e**-input) + 1)**-1
        return output

    def forward_prop(self):
        for index0 in range(1, 4):
            self.layers[index0] = self.bias[index0 - 1]

        for index0 in range(1, 4):
            for index1 in range(9):
                for index2 in range(9):
                    self.layers[index0][index1] = self.layers[index0][index1] + self.layers[index0 - 1][index2] * \
                                                  self.weights[index0 - 1][index1][index2]

            self.layers = self.sigmoid_activation_function(self.layers)

    def eta(self):
        return np.cosh(self.score ** score_adaptivity) ** -1

    def spot_of_largest_output_value(self):
        return np.argmax(self.layers[3, :])


class Game:
    def __init__(self):
        self.game_board = np.repeat(0, 9)
        self.is_cheated = False
        self.player_winner = np.repeat(False, 2)

    def game_over(self, sign=np.array([1, -1])):
        for player_number in range(2):
            for location_on_board in range(3):
                if self.game_board[3*location_on_board] == self.game_board[3*location_on_board + 1] == self.game_board[3*location_on_board + 2] == sign[player_number]:
                    self.player_winner[player_number] = True
                    break

            for location_on_board in range(3):
                if self.game_board[location_on_board] == self.game_board[location_on_board + 3] == self.game_board[location_on_board + 6] == sign[player_number]:
                    self.player_winner[player_number] = True
                    break

            for location_on_board in range(2):
                if self.game_board[2*location_on_board] == self.game_board[4] == self.game_board[-2*location_on_board + 8] == sign[player_number]:
                    self.player_winner[player_number] = True
                    break

    def player_0s_turn(self, input):
        if self.game_board[input] == 0:
            self.game_board[input] = 1

        else:
            self.is_cheated = True

    def player_1s_turn(self, input):
        if self.game_board[input] == 0:
            self.game_board[input] = -1

        else:
            self.is_cheated = True

    def game_reset(self):
        self.game_board = np.repeat(0, 9)
        self.player_winner = np.repeat(False, 2)
        self.is_cheated = False


neural_network = np.array([NeuralNetwork()])

for network_number in range(int(number_of_networks) - 1):
    neural_network = np.insert(neural_network, 0, NeuralNetwork())

game = Game()


def evaluate_scores():
    for index0 in range(number_of_networks):
        neural_network[index0].score = 0

    for index0 in range(int(number_of_networks)):
        for index1 in range(int(number_of_networks)):
            if index0 != index1:
                for index2 in range(9):
                    neural_network[index0].layers[0][index2] = game.game_board[index2]

                neural_network[index0].forward_prop()
                game.game_board[neural_network[index0].spot_of_largest_output_value()] = 1

                for index2 in range(4):
                    indices = np.array([index1, index0, -1, 1])

                    for players_turn in range(2):
                        for index3 in range(9):
                            neural_network[indices[players_turn]].layers[0][index3] = indices[players_turn + 2] * \
                                                                                      game.game_board[index3]

                        neural_network[indices[players_turn]].forward_prop()
                        game.player_0s_turn(neural_network[indices[players_turn]].spot_of_largest_output_value())

                        if game.is_cheated:
                            neural_network[indices[(players_turn + 1) % 2]].score = neural_network[indices[
                                (players_turn + 1) % 2]].score + 1
                            break

                        else:
                            game.game_board[neural_network[indices[players_turn]].spot_of_largest_output_value()] = \
                                indices[players_turn + 2]

                        game.game_over()

                        if game.player_winner[players_turn]:
                            neural_network[indices[players_turn]].score = \
                                neural_network[indices[players_turn]].score + 1
                            break

                    if game.is_cheated or game.player_winner[0] or game.player_winner[1]:
                        break

            else:
                pass

            game.game_reset()


def relevance_of_network(input):
    divisor = 0

    for index0 in range(int(number_of_networks)):
        divisor = divisor + neural_network[index0].score

    if divisor == 0:
        divisor = 1

    return neural_network[input].score/divisor


def new_set_of_neural_networks():
    scores = np.array([neural_network[0].score])

    for index0 in range(1, int(number_of_networks)):
        scores = np.insert(scores, index0, neural_network[index0].score)

    scores = np.argsort(scores)

    for index0 in range(int(number_of_networks * ratio_of_reset_networks)):
        for index1 in range(3):
            for index2 in range(9):
                for index3 in range(9):
                    neural_network[scores[index0]].weights[index1][index2][index3] = \
                        neural_network[scores[int(number_of_networks) - 1]].weights[index1][index2][index3] + \
                        2*radius_of_random_generation * (np.random.rand() - 0.5)

    for index0 in range(int(number_of_networks * ratio_of_reset_networks)):
        for index1 in range(3):
            for index2 in range(9):
                neural_network[scores[index0]].bias[index1][index2] = \
                    neural_network[scores[int(number_of_networks) - 1]].bias[index1][index2] + \
                    2*radius_of_random_generation * (np.random.rand() - 0.5)

    for index0 in range(int(number_of_networks * ratio_of_reset_networks), int(number_of_networks) - 1):
        for index1 in range(3):
            for index2 in range(9):
                for index3 in range(9):
                    for index4 in range(int(number_of_networks * ratio_of_reset_networks), int(number_of_networks)):
                        neural_network[scores[index0]].weights[index1][index2][index3] = \
                            neural_network[scores[index0]].weights[index1][index2][index3] + \
                            learning_rate * neural_network[scores[index0]].eta() * \
                            relevance_of_network(index4) * \
                            (neural_network[scores[index4]].weights[index1][index2][index3] - \
                             neural_network[scores[index0]].weights[index1][index2][index3])

    for index0 in range(int(number_of_networks * ratio_of_reset_networks), int(number_of_networks) - 1):
        for index1 in range(3):
            for index2 in range(9):
                for index3 in range(int(number_of_networks * ratio_of_reset_networks), int(number_of_networks)):
                    neural_network[scores[index0]].bias[index1][index2] = \
                        neural_network[scores[index0]].bias[index1][index2] + \
                        learning_rate * neural_network[scores[index0]].eta() * \
                        relevance_of_network(index3) * \
                        (neural_network[scores[index3]].bias[index1][index2] - \
                         neural_network[scores[index0]].bias[index1][index2])


def neural_network_versus_human():
    global flip_turn

    blank_screen = np.repeat(' ', 200)

    scores = np.array([neural_network[0].score])

    for index0 in range(1, int(number_of_networks)):
        scores = np.insert(scores, index0, neural_network[index0].score)

    scores = np.argsort(scores)

    game.game_reset()

    if flip_turn is False:
        human_input = input()

        for index0 in range(4):
            if human_input == 'f':
                flip_turn = True
                break

            game.game_board[int(human_input)] = 1

            print(blank_screen)
            print(game.game_board[0], game.game_board[1], game.game_board[2])
            print(game.game_board[3], game.game_board[4], game.game_board[5])
            print(game.game_board[6], game.game_board[7], game.game_board[8])

            for index1 in range(9):
                neural_network[scores[int(number_of_networks) - 1]].layers[0][index1] = -1 * game.game_board[index1]

            neural_network[scores[int(number_of_networks) - 1]].forward_prop()
            game.player_0s_turn(neural_network[scores[int(number_of_networks) - 1]].spot_of_largest_output_value())

            if game.is_cheated:
                print('Cheat!')
                break

            else:
                game.game_board[neural_network[scores[int(number_of_networks) - 1]].spot_of_largest_output_value()] = -1

            game.game_over()

            if game.player_winner[1]:
                print('Artificial Neural Network wins')
                break

            print(blank_screen)
            print(game.game_board[0], game.game_board[1], game.game_board[2])
            print(game.game_board[3], game.game_board[4], game.game_board[5])
            print(game.game_board[6], game.game_board[7], game.game_board[8])

            human_input = input()

            if human_input == 'f':
                flip_turn = True
                break

            game.game_board[int(human_input)] = 1

            game.game_over()

            if game.player_winner[0]:
                print('human wins!')
                break

        if game.player_winner[0] == game.player_winner[1] and game.is_cheated == flip_turn:
            print('draw!')

    else:
        for index1 in range(9):
            neural_network[scores[int(number_of_networks) - 1]].layers[0][index1] = game.game_board[index1]

        neural_network[scores[int(number_of_networks) - 1]].forward_prop()
        game.game_board[neural_network[scores[int(number_of_networks) - 1]].spot_of_largest_output_value()] = 1

        for index0 in range(4):
            print(blank_screen)
            print(game.game_board[0], game.game_board[1], game.game_board[2])
            print(game.game_board[3], game.game_board[4], game.game_board[5])
            print(game.game_board[6], game.game_board[7], game.game_board[8])

            human_input = input()

            if human_input == 'f':
                flip_turn = False
                break

            game.game_board[int(human_input)] = -1

            game.game_over()

            if game.player_winner[1]:
                print('human wins!')
                break

            print(blank_screen)
            print(game.game_board[0], game.game_board[1], game.game_board[2])
            print(game.game_board[3], game.game_board[4], game.game_board[5])
            print(game.game_board[6], game.game_board[7], game.game_board[8])

            for index1 in range(9):
                neural_network[scores[int(number_of_networks) - 1]].layers[0][index1] = game.game_board[index1]

            neural_network[scores[int(number_of_networks) - 1]].forward_prop()
            game.player_1s_turn(neural_network[scores[int(number_of_networks) - 1]].spot_of_largest_output_value())

            if game.is_cheated:
                print('Cheat!')
                break

            else:
                game.game_board[neural_network[scores[int(number_of_networks) - 1]].spot_of_largest_output_value()] = 1

            game.game_over()

            if game.player_winner[0]:
                print('Artificial Neural Network wins')
                break

        if game.player_winner[0] == game.player_winner[1] and game.is_cheated != flip_turn:
            print('draw!')


def main():
    try:
        for iterate in range(number_of_iterations):
            evaluate_scores()
            new_set_of_neural_networks()

            print(100*(iterate/number_of_iterations), '%')
    except Exception as ex:
        print("Exception: {}".format(ex))
        scores = np.array([neural_network[0].score])

        for index0 in range(1, int(number_of_networks)):
            scores = np.insert(scores, index0, neural_network[index0].score)

        scores = np.argsort(scores)

        print('\nWeights: ', '\n\n', neural_network[scores[number_of_networks - 1]].weights, '\n\nbias: \n\n',
              neural_network[scores[number_of_networks - 1]].bias)


    scores = np.array([neural_network[0].score])

    for index0 in range(1, int(number_of_networks)):
        scores = np.insert(scores, index0, neural_network[index0].score)

    scores = np.argsort(scores)

    print('\nWeights: ', '\n\n', neural_network[scores[number_of_networks - 1]].weights, '\n\nbias: \n\n', neural_network[scores[number_of_networks - 1]].bias)

    while True:
        neural_network_versus_human()


if __name__ == "__main__":
    main()
