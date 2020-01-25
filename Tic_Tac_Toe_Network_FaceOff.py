import numpy as np


randoms_score = 0


class NeuralNetwork:
    def __init__(self):
        self.layers = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.bias = 2*(np.random.rand(3, 9) - 0.5)
        self.weights = 2*(np.random.rand(3, 9, 9) - 0.5)
        self.score = 0

    def sigmoid(self, input):
        output = ((np.e**-input) + 1)**-1
        return output

    def forward_prop(self):
        print(self.weights)
        for index0 in range(1, 4):
            for index1 in range(9):
                self.layers[index0][index1] = self.bias[index0 - 1][index1]

        for index0 in range(1, 4):
            for index1 in range(9):
                for index2 in range(9):
                    self.layers[index0][index1] = self.layers[index0][index1] + self.layers[index0 - 1][index2] * \
                                                  self.weights[index0 - 1][index1][index2]

            self.layers = self.sigmoid(self.layers)

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


game = Game()
neural_network = NeuralNetwork()

neural_network.weights = np.array([[[-24.23351795, -22.995247, -30.54636913, 77.44755244, -4.92014807,
                                     -9.74113621, 44.89954423, -31.27811241, 8.99929491],
                                    [-52.1585649, -2.29397991, -56.75206535, -7.70462149, 10.04902517,
                                     -43.90253467, 19.5190761, -22.71808016, -27.10451808],
                                    [-0.41119395, 3.61408999, 17.88253223, 24.90092712, -28.29190369,
                                     37.67844472, -27.3869185, 89.13269115, -92.33929196],
                                    [57.63160047, -21.24843793, 7.14705938, -20.23277258, -10.5161785,
                                     29.56202926, -50.09985576, -97.27002317, -1.61433688],
                                    [158.8705412, -1.98306034, 23.91297447, -89.30272614, 26.68410517,
                                     -62.04210065, -93.46322864, -19.10744917, 45.10408609],
                                    [-8.56244233, -20.14975678, -23.41466719, -55.97590892, -58.74570733,
                                     11.26194148, 1.8436841, -46.78678268, -27.09327134],
                                    [-24.29884455, 21.49700806, -96.16489939, 13.70950438, 29.46595612,
                                     -97.1978879, 9.12302008, -13.26483, -9.80480112],
                                    [4.29324474, -3.06910763, 162.67561431, 54.15424071, 30.10575605,
                                     -37.62446951, -63.01367441, -15.90373777, -45.90035323],
                                    [7.04440432, -7.42682683, -15.24153088, 59.63818434, 75.40951294,
                                     -30.93964715, -3.87373956, 17.29440403, 14.1803095]],
                                   [[-58.61300929, -13.14635306, 5.89810519, -21.12305326, 28.09790308,
                                     108.57708927, -22.49285843, 23.08622342, 42.1674317],
                                    [24.60495199, -34.42621678, -58.57636378, 28.00578706, 45.87765798,
                                     -11.9803429, 32.1212776, -73.07339185, 18.29960825],
                                    [4.53279947, 6.61789345, -40.53052019, 17.01696426, 13.42016919,
                                     9.31006433, 0.37516248, 14.30464002, -23.66414604],
                                    [3.636514, 16.30155996, -9.30271505, -21.22592004, -40.23655496,
                                     28.42711293, 25.66392778, 43.11581211, 40.47126064],
                                    [-95.3214853, 38.3897172, 25.02920989, 61.40279925, -45.92887437,
                                     54.30352339, 11.4959652, -23.83934933, 36.45939759],
                                    [12.69497006, -21.07849481, -63.13640988, -47.76088455, 45.04757313,
                                     -57.51633667, 69.25299746, -23.48281305, 47.62349205],
                                    [-23.15437502, -5.03468811, -7.05903603, 18.41430792, -35.11049899,
                                     74.04849792, -6.5238235, -25.70163964, 39.00420398],
                                    [-1.63244853, -10.32818626, -40.60788734, -25.74487552, -82.4204106,
                                     26.52886786, 3.0867573, 14.05446488, 10.86005481],
                                    [-130.66844119, 15.54413327, -44.49476735, -30.78459936, -2.7746936,
                                     -3.62375925, 61.28705784, -58.87599594, -7.23849362]],
                                   [[-35.68215678, -28.09351057, -27.08092794, -18.98678485, -27.10481729,
                                     -34.94211471, 97.38173872, 1.91715336, -44.06995669],
                                    [-42.12136788, -37.21860601, 17.60473065, 24.92260906, 54.97170184,
                                     -52.86968353, -12.50161238, 49.08723102, -14.5492629],
                                    [-3.53399793, 17.2492209, -34.42421646, -0.8783846, -57.09212247,
                                     -7.44742395, 51.03002507, 2.79062229, -58.1181504],
                                    [-18.34205288, 0.36005508, -11.15324189, 2.94500426, -63.25079227,
                                     32.11094618, 72.30030855, 67.71344762, 10.22188414],
                                    [18.62581911, -27.90973718, -37.70589981, -17.1147997, -19.7064243,
                                     8.04309266, -18.70169897, -50.20803471, 43.93393998],
                                    [59.34669837, -40.23824842, 28.36207468, -116.08916392, 17.23346181,
                                     23.67877858, -38.23028607, -1.69331339, 73.17039958],
                                    [-25.51548231, 37.74373604, 15.97588266, -4.77770629, 20.56347195,
                                     4.48093496, 31.27782608, -46.94326773, 62.93690454],
                                    [-21.24026971, 59.13021909, -62.60010723, -36.65297519, 34.50954784,
                                     23.4605394, 5.25714356, -3.83576087, -31.62695052],
                                    [24.47057811, 71.90172217, -12.89993809, 10.68411467, 36.48345986,
                                     83.75456719, 21.78522301, -2.61457858, -71.87389445]]])


neural_network.bias = np.array([[29.3305765-4, 9.31276876, 39.46678994, -32.41922853, 19.83892312,
                                    38.45028143, -38.26741718, 16.82474353, 7.31872769],
                                   [-14.6599455, 7.4514441, -17.43134322, -61.14992259, 10.2706292,
                                    -23.12026647, -4.27212252, 88.29489484, 28.73065923],
                                   [-14.53815913, 23.26766797, -71.15844196, -47.17290012, 83.85359057,
                                    -90.13779717, 8.02836395, 20.46580834, 1.85432295]])


def versus():
    global randoms_score

    game.game_reset()

    for index0 in range(9):
        neural_network.layers[0][index0] = game.game_board[index0]

    neural_network.forward_prop()
    game.game_board[neural_network.spot_of_largest_output_value()] = 1

    for index0 in range(4):
        output = np.random.randint(9)
        game.player_1s_turn(output)

        if game.is_cheated:
            neural_network.score = neural_network.score + 1
            break

        else:
            game.game_board[output] = -1

        game.game_over()

        if game.player_winner[1]:
            randoms_score = randoms_score + 1
            break

        for index1 in range(9):
            neural_network.layers[0][index1] = game.game_board[index1]

        neural_network.forward_prop()
        game.player_0s_turn(neural_network.spot_of_largest_output_value())

        if game.is_cheated:
            randoms_score = randoms_score + 1
            break

        else:
            game.game_board[neural_network.spot_of_largest_output_value()] = 1

        game.game_over()

        if game.player_winner[0]:
            neural_network.score = neural_network.score + 1
            break

    game.game_reset()

    output = np.random.randint(9)
    game.game_board[np.random.randint(9)] = 1

    for index0 in range(4):
        for index1 in range(9):
            neural_network.layers[0][index1] = -1*game.game_board[index1]

        neural_network.forward_prop()
        game.player_1s_turn(neural_network.spot_of_largest_output_value())

        if game.is_cheated:
            randoms_score = randoms_score + 1
            break

        else:
            game.game_board[neural_network.spot_of_largest_output_value()] = -1

        game.game_over()

        if game.player_winner[1]:
            neural_network.score = neural_network.score + 1
            break

        output = np.random.randint(9)
        game.player_0s_turn(output)

        if game.is_cheated:
            neural_network.score = neural_network.score + 1
            break

        else:
            game.game_board[output] = 1

        game.game_over()

        if game.player_winner[0]:
            randoms_score = randoms_score + 1
            break


def main():
    for iterate in range(100):
        versus()

    print('The Neural Networks Score: ', neural_network.score, '\n\n', '"Randoms" score: ', randoms_score)


if __name__ == "__main__":
    main()
