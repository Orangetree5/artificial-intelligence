from PIL import Image
from numpy import complex
from sympy import *
import numpy as np
from logging import basicConfig, getLogger



Training_Set = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])



class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation



def sigmoid(input):
    output = ((np.e**-input) + 1)**-1

    return output



starting_weight_array = np.array([np.random.random_sample(), np.random.random_sample()], dtype='f')
starting_bias = np.random.random_sample()

Neuron_0 = 0
Neuron_1 = 0
Neuron_2 = Neuron(starting_weight_array, starting_bias, 0)
Neuron_3 = Neuron(starting_weight_array, starting_bias, 0)
Neuron_4 = Neuron(starting_weight_array, starting_bias, 0)



Layer_0 = np.array([Neuron_0, Neuron_1], dtype='f')
Layer_1 = np.array([Neuron_2.activation, Neuron_3.activation], dtype='f')



All_Dimensions = np.array([Neuron_2.weights[0], Neuron_2.weights[1], Neuron_3.weights[0], Neuron_3.weights[1], Neuron_4.weights[0], Neuron_4.weights[1], Neuron_2.bias, Neuron_3.bias, Neuron_4.bias], dtype='f')



weight_0 = Symbol('weight_0')
weight_1 = Symbol('weight_1')

weight_vector_neuron_2 = np.array([weight_0, weight_1])

weight_2 = Symbol('weight_2')
weight_3 = Symbol('weight_3')

weight_vector_neuron_3 = np.array([weight_2, weight_3])

weight_4 = Symbol('weight_4')
weight_5 = Symbol('weight_5')

weight_vector_neuron_4 = np.array([weight_4, weight_5])

bias_2 = Symbol('bias_2')
bias_3 = Symbol('bias_3')
bias_4 = Symbol('bias_4')

Activation_2 = sigmoid(Layer_0.dot(weight_vector_neuron_2) + bias_2)
Activation_3 = sigmoid(Layer_0.dot(weight_vector_neuron_3) + bias_3)

Function_Layer = np.array([Activation_2, Activation_3])

Activation_4 = sigmoid(Function_Layer.dot(weight_vector_neuron_4) + bias_4)



def forward_prop():
    Neuron_2.activation = sigmoid(Layer_0.dot(Neuron_2.weights) + Neuron_2.bias)
    Neuron_3.activation = sigmoid(Layer_0.dot(Neuron_3.weights) + Neuron_3.bias)

    Neuron_4.activation = sigmoid(Layer_1.dot(Neuron_4.weights) + Neuron_4.bias)

    return Neuron_4.activation



def gradient():
    Dimension_0 = 0
    Dimension_1 = 0
    Dimension_2 = 0
    Dimension_3 = 0
    Dimension_4 = 0
    Dimension_5 = 0
    Dimension_6 = 0
    Dimension_7 = 0
    Dimension_8 = 0

    activation_neuron_04 = np.array([0, 0, 0, 0], dtype='f')

    for training_set_index in range(0, 4):
        Neuron_0 = Training_Set[training_set_index][0]
        Neuron_1 = Training_Set[training_set_index][1]
        Layer_0 = np.array([Neuron_0, Neuron_1], dtype='f')

        forward_prop()

        activation_neuron_04[training_set_index] = Neuron_4.activation

        Cost_Function_Variable_Form = (Activation_4 - Training_Set[training_set_index][2])**2



        Dimension_0 = Dimension_0 + float(Cost_Function_Variable_Form.diff(weight_0)*0.1)
        Dimension_1 = Dimension_1 + float(Cost_Function_Variable_Form.diff(weight_1)*0.1)
        Dimension_2 = Dimension_2 + float(Cost_Function_Variable_Form.diff(weight_2)*0.1)
        Dimension_3 = Dimension_3 + float(Cost_Function_Variable_Form.diff(weight_3)*0.1)
        Dimension_4 = Dimension_4 + float(Cost_Function_Variable_Form.diff(weight_4)*0.1)
        Dimension_5 = Dimension_5 + float(Cost_Function_Variable_Form.diff(weight_5)*0.1)
        Dimension_6 = Dimension_6 + float(Cost_Function_Variable_Form.diff(bias_2)*0.1)
        Dimension_7 = Dimension_7 + float(Cost_Function_Variable_Form.diff(bias_3)*0.1)
        Dimension_8 = Dimension_8 + float(Cost_Function_Variable_Form.diff(bias_4)*0.1)

    gradient_vector = np.array([Dimension_0, Dimension_1, Dimension_2, Dimension_3, Dimension_4, Dimension_5, Dimension_6, Dimension_7, Dimension_8], dtype='f')

    print(Training_Set[0][0], ', ', Training_Set[0][1], ' => ', Training_Set[0][2], ': ', activation_neuron_04[0], '   ', Training_Set[1][0], ', ', Training_Set[1][1], ' => ', Training_Set[1][2], ': ', activation_neuron_04[1], '   ', Training_Set[2][0], ', ', Training_Set[2][1], ' => ', Training_Set[2][2], ': ', activation_neuron_04[2], '   ', Training_Set[3][0], ', ', Training_Set[3][1], ' => ', Training_Set[3][2], ': ', activation_neuron_04[3])

    return gradient_vector


def Back_Prop(All_Dimensions):
    print('stuff: ', All_Dimensions, 'stuff: ', Neuron_4.weights[0])

    All_Dimensions = All_Dimensions - gradient()

    print('afterculcu: ', All_Dimensions[4], 'afterculcu: ', Neuron_4.weights[0])

    Neuron_2.weights[0] = All_Dimensions[0]
    Neuron_2.weights[1] = All_Dimensions[1]
    Neuron_3.weights[0] = All_Dimensions[2]
    Neuron_3.weights[1] = All_Dimensions[3]
    print(type(Neuron_4.weights[0]))
    print(type(All_Dimensions[3]))
    Neuron_4.weights[0] = All_Dimensions[4]
    Neuron_4.weights[1] = All_Dimensions[5]
    Neuron_2.bias = All_Dimensions[6]
    Neuron_3.bias = All_Dimensions[7]
    Neuron_4.bias = All_Dimensions[8]

    return 0



if __name__=="__main__":
    for x in range(0, 100000):
        gradient()
        Back_Prop(All_Dimensions)