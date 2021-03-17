from scripts import NN
from scripts import io
import pytest
import numpy
import random as rand

@pytest.fixture
def simple_network():
    """A very simple neural network that is useful for tests below"""
    setup = [8,3,1]
    alpha = 0.5
    lamba = 0 
    simple_NN = NN.NeuralNetwork(setup,NN.activation,alpha,lamba)
    
    return simple_NN

def test_pos_import():
    """Tests if the correct number of k-mers are imported and if the last one is the last one in the file"""
    pos_list = io.import_positives('data/rap1-lieb-positives.txt')
    
    assert len(pos_list) == 137
    assert pos_list[136] == 'ACACCCATACACCAAAC'
    
    
def test_neg_import():
    """Tests if a given k-mer in the negative list is the correct size. Specific ones can't be tested because
    they are randomly selected"""
    neg_list = io.import_negatives('data/yeast-upstream-1k-negative.fa',17)
    
    assert len(neg_list[0]) == 17

def test_one_hot():
    """This function tests if one hot encoding produces the correct encoding for a simple example string"""
    test_sequence = ["ATGC"]
    result = io.one_hot_encode(test_sequence)
    result = result[0]
    
    assert numpy.array_equal(result, [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])

    
def test_activation(simple_network):
    """This function test if the activation function and its derivative have the correct braod behavior"""
    assert simple_network.act_f(0) == 0.5
    assert simple_network.act_f(-1) < 0.5
    assert simple_network.act_f(1) > 0.5
    assert simple_network.sigmoid_deriv(-1) > 0
    assert simple_network.sigmoid_deriv(0)  > 0
    assert simple_network.sigmoid_deriv(1)  > 0
    

def test_feed_forward(simple_network):
    """This functoin tests that the matrix multiplication for feedforward works with very simple matrices and biases"""
    #manually set matrices and biases
    simple_network.edge_matrices[0] = numpy.ones((3,8))
    simple_network.edge_matrices[1] = numpy.ones((1,3))
    simple_network.biases[0] = numpy.ones((3))
    simple_network.biases[1] = numpy.ones((1))
    #use simple input
    simple_network.get_single_input([1,1,1,1,1,1,1,1])
    simple_network.feedforward()
    assert numpy.isclose(simple_network.layer_a[1][0],0.9820072)
    
def test_backprop():
    """This function test if backprop produces matrices in the correct list orientation and dimensions. Doesn't
    actually check if the values are correct given the complex calculations. Does so in the auto-encoder context."""
    setup = [8,3,8]
    alpha = 0.5
    lamba = 0 
    simple_NN = NN.NeuralNetwork(setup,NN.activation,alpha,lamba)
    simple_NN.get_single_input([1,0,0,0,0,0,0,0])
    simple_NN.feedforward()
    Ws, bs = simple_NN.backprop(simple_NN.edge_matrices,simple_NN.biases,simple_NN.input_layer,simple_NN.layer_z,simple_NN.layer_a)
    
    #test the dimensions of the returned partial W and partial B matrices
    assert len(Ws[0]) == 3
    assert len(Ws[1]) == 8
    assert len(Ws[0][0]) == 8
    assert len(Ws[1][0]) == 3
    assert len(bs[0]) == 3
    assert len(bs[1]) == 8

    
def test_simple_classify(simple_network):
    """This function tests if the network can classify a bit in the first half of the input as 0 and a bit in the second half as 1. It also tests to see
    if cost is minimized to atleast some degree over the course of training."""
    #basic input parameteres
    epoch_number = 5000
    cost_list = []
    batch_size = 1
    input_size   = 8
    output_size  = 1
    #manual training protocol
    for j in range(epoch_number):
        training_list = []
        answer_list   = []
        #batches manually fed in
        for k in range(batch_size):
            #training vectors have 1 bit randomly on
            training_batch = numpy.zeros((input_size))
            training_answer = numpy.zeros((output_size))
            rand_index = int(numpy.ceil(rand.random()*8)-1)
            training_batch[rand_index] = 1
            training_list.append(training_batch)
            #correct answer is 1 if the bit is flipped after the 3rd index
            if rand_index > 3: 
                training_answer[0] = 1
            answer_list.append(training_answer)
        #train the network    
        simple_network.get_training_set(training_list,answer_list)
        cost_list.append(simple_network.batch_descent())
    
    assert cost_list[0] > cost_list[len(cost_list)-1]
    assert simple_network.predict([0,0,0,0,0,0,0,1]) > 0.75
    assert simple_network.predict([1,0,0,0,0,0,0,0]) < 0.25


    
    