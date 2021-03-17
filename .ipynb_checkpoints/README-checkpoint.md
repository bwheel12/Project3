# Project 3 - Neural Networks
## Due 03/19/2021

![BuildStatus](https://github.com/bwheel12/Project3/workflows/HW3/badge.svg?event=push)

All the code and prose to answer the questions for the assignment are contained in Benjamin_Wheeler_Project3.ipynb. The cells can be run begining to end will produce the notebook as is. The order they are called is somewhat out of order as things were tweaked (plots, etc.) in the final submision. The genetic algorithm takes a long time to run, so those cells were never re-run. \_\_main\_\_.py and \_\_init\_\_.py are unused. 
The predictions for the unknown test set are in final_test_scores.tsv in the main directory. 


### testing
Testing is as simple as running
```
python -m pytest test/*
```
from the root directory of this project.

## Classes and Functions within NN.py

## Class NeuralNetwork
This class implements a fully connected neural network. Setup should be a vector where the first entry indicates the input 
layer size, the last the output layer size, and for any given number of entries in between the size of the respective hidden layers. 
A function should be passed as act_f which is the activation function. alpha is the learning rate should be a scaler 0 to 1. 
Lamba should be a very small scaler.

### Functions in NeuralNetwork

#### get_single_input(self,input_layer):
This function takes a single input and assigns it to the input layer
```
        Parameters: input layer, a vector the same length as the indicated input layer
        returns: none
    
```

#### get_training_set(self,inputs,answers):
This function takes a list of training examples and answers and stores them for training
```
        parameters: inputs, a list of one hot(or other wise numerically) encoded training examples
        answers: a list of answers to which the output layer should be compared. should be numerical and size match output layer
```

#### feedforward(self):
This function does the matrix multiplication to translate the input layer into the output via passing through
fully connected layers with the defined activation function. No parameters given acts entirely on self variables.

#### batch_descent(self):
This function uses the training inputs and answers to compare the current feedforward answer to the desired output.
It then uses gradient descent to update the network weights to train the network. No parameters are given as it works 
entirely on self variables

#### test_backprop(self):
This function only does the gradient calculation step within back prop. really useful for debugging. Kept altough no longer needed in final implementation

#### backprop(self,edge_matrices,biases,correct,layer_zs,layer_as):
This function does the gradient calculations. Might not fully encompas what might be considered 'backprop' but the essential difficult step. And
it is too late to change the name now. It takes all the essential components of the network as they typically are. It returns the partial gradient values
for a given correct answer.

```
        Parameters: edge_matrices: list of scaler matrices with dimensions corresponding to the pairs of the layers of the network
                    biases: list of scaler scaler vectors corresponding to the size of each non-input, layer
                    correct: should be a scaler vector of the same size as the output layer
                    layer_zs: list of scaler vectors indicating the summed by not actiated values for each layer
                    layer_as: list of scaler vectors indicating the "activation" value for each node in each layer
                    
        returns:    partial_Ws: list of scaler matrices with the same dimensions as edge_matrices indicating the gradient values
                    partial_bs: list of scaler vectors with the same dimensions of biases indicating the gradient values
                    
```


#### derivatives_vector(self,x):
This function is useful for backprop above. It takes a vector of numerical values and calculates the sigmoidal derivative
value of each and returns it in a similar vector


```
        Parameters: x: a vector of scaler values
        returns: deriv_vect: a vector of scalcer values of the sigmoid derivative value at the corresponding value in x
        
```

#### act_vector(self,x):
This function is useful for feedforward. It takes a vector of numerical values and calculates the activation function value 
of each element and returns it in a similar vector

```
        Parameters: x, a vector of scaler values
        Returns: act_vect: a vector of scaler values with the value of the activation function applied to the corresponding value in x
```

#### train(self,epoch_number,batch_size, whole_set, answer_set):
This function trains the network by interating through a training set containing negative and positive examples and a corresponding answer set
this function expects ATCG encoding as it does the one hot encoding call. This is also where batch size gets used to break the training set up into
smaller chunks with which the weights get updated. I.e. at the end of the k loop the weights are actually updated

```
        Parameters: epoch_number: integer the number of times to iterate through the whole training set
                    batch_size: the number of examples to go through before updating weights
                    whole_set: all the training examples a list of ATCG encoded positive and negative examples
                    answer_set: a correpsonding list of the 'correct' answer for each example
        returns:   cost_list: a list of scaler values corresponding to the loss after each batch of samples
```

#### predict(self,input_layer):
This function takes a single input and gets the network prediction by feeding it forward through. The prediction is returned.

```
        Parameters: input_layer: the input to be classified or predicted. Can be any size, but needs to be the same size as the initialized input layer
        returns:    answer: the value of the output layer indicative of the networks classification of the example. For sigmoid activation function [0,1]
        
```


#### sigmoid_deriv(self,y):
This function is the derivative of the sigmoidal activation function

```
        Parameters: y any scaler value
        Returns the value of the sigmoidal derivative for the value of y
```

### Functions outside the NeuralNetwork class but useuful for the project broadly

#### activation(x):
This function implements a sigmoid activation function

```
    parameters: x float
    returns:    f_of_x float bounded 0 to 1
```


#### k_fold_validation(train_set,answer_set,epoch_number,batch_number,setup,alpha,lamba,k):
This function will provide k-fold cross validation on a given training set, with a given model. In this case the parameters needed create the NN are passed
and each validation loop will create a new temporary network with which to train and validate. For each temporarily model it will evaluate on the held out validation
set and calculate True Positive Rates and False Positive Rates at a series of threshold values between 0 and 1. It returns these True Positive Rates and False Positive 
rates as nested lists with a list within each for each validation cycle.

```
    Parameters: train_set: a whole set of examples to train on both positive and negative ATCG encoded
                answer_set: a corresponding list of proper classifications
                epoch_number: int, the number of times to iterate through the whole data set
                batch_number: int, the number of samples to calculate the partial derivatives for before updating weights
                setup: int vector, indicates the architecture of the network
                alpha; scaler [0,1] the learning rate
                lamba: scaler [0,1] the weight decay constant
                k: int, the number of validation iterations to perform. also the number of chunks the data set is split into
    Returns: roc_FPRs: a nest list of scaler values corresponding to the false positive rate for various scoring thresholds for each validation iteration
             roc_TPRs: a nest list of scaler values corresponding to the true positive rate for various scoring thresholds for each validation iteration
```

#### roc_auc_list(roc_TPRs,roc_FPRs):
This function expects lists of lists of TPRs and FPRs (ie at least 2 sets per TPR and FPR). With those lists it calculates the area under the 
curve of each corresponding ROC curve.

```
    Parameters: roc_TPRs: nested list of scaler values corresponding to True positive rate produced by k_fold_validation
                roc_FPRs: nested list of scaler values corresponding to False positive rate produced by k_fold validation
    Returns: roc_AUCs: list of scalers, the area under each ROC curve produced by k-fold validation
```

#### genetic_optimization(train_set,answer_set,generation_num,pop_num,cross_rate,mutation_rate, centers):
This function implements a genetic algorithm to optimize the network, this assumes a network with a 68 node input layer,
a 1 node output layer, and 2  hidden layers. I will seek to optmize the size of each hidden layer, the learning rate, the rate of weight decay, 
the number of epochs to perform, and the batch size. I will use the average auROC from k-fold cross validation to evaluate each population member.
pop_num should be even. 

```
    Parameters: train_set: the whole training set of positive and negative examples, list of ATCG encoded examples
                answer_set: list of scaler values, the corresponding correct labels
                generation_num: int, the number of generations to "evolve"
                pop_num: int, the number of individuals in each generation
                cross_rate: scalcer [0,1] the rate at which given traits should be swapped between individuals
                mutation_rate: scaler [0,1] the rate at which give traits should be mutated
                centers: numerical vector containing the (hyper)parameters of the model to be optimized. Specifically the center value in which
                initial values should be randomly generated from for the first generation
    returns: best_individual: numerical vector of parameter set from the highest scoring individual of the last generation
            gen_avg_score: scaler vector, the mean of the scores for each generation
            gen_max_score: scaler vector, the max score from each generation
            generations: nested list, all of the "individual" parameter sets generated

```

#### check_individual(individual,train_len):
This function is needed to keep the parameters manipulated by the genetic algorithm above within acceptable contexts. Accepts a population
individual in the paradigm as above and train_len which is the length of the training set. This is important because it is the limit for 
batch size. It generally fixes errors beyond a bound by reseting it to an acceptable boundary condition

```
    Parameters: Individual: numeric vector, Indicates the (hyper)parameter settings for a given "individual"
                train_len: int, the length of the training set
    Returns:   individual: numeric vector, the corrected parameter settings for the bounds of the model
```

#### score_generation(generation,train_set,answer_set):
This function scores a generation from the genetic algorithm by calculating the auROC for the given parameters of an individual.
It is hard coded for k=2 cross validation to save time in scoring generations. Essentially it is the standard initial,train,evaluate for
k-fold cross validation just looped over many temporary networks. Requires a list of 'individuals' and the training and answer set. The training set should be
in ATCG encoding. It returns a vector of the scores in the order of the generation individuals.

```
    Parameters: generation: list of numerical vectors: the individuals to score from the genetic algorithm
                train_set: list of ATCG encoded examples, the whole training set both positive and negative examples
                answer_set: numerical list, the corresponding correct answer list
    Returns:   generation_scores: scaler list, the scores for the generation from the mean auROC of k-fold cross validation
```



## Functions within io.py

#### import_positives(import_path):
This function expects the path to a file to import DNA sequences for the positve examples and provides a list 
where each entry is a string from a line in the origianl file

```
    Parameters: import_path: string, indicates path to txt file with k-mer list
    Returns: positive list: a list of ATCG encoded examples from the file
```

#### import_negatives(import_path,k_mer):
This function takes the path to the 1K length fasta upstream sequences
it randomly selects a k_mer length string from a random sequence line from each gene

```
    Parameters: import_path: string, indicates path to read the sequences from
                k_mer: int, the length of the k-mers to return
    Returns: negative list: list of ATCG encoded k-mers from fasta sequences
```

#### one_hot_encode(k_mer_list):
This function takes a list of k-mers A,G,C,T and returns a list where each entry is a 1-hot encoded version of the k-mer

```
    Parameters: k_mer_list: a list of ATCG enocded examples 
    Returns:    one_hot_encoded: a list of one hot encoded examples
    
```


####  shuffle_concat(pos_list,neg_list,neg_scale):
This function creates a single training set from individual positive and negative example lists. Must provide the example lists in ATCG encoding. Neg scale is 
an integer that is the number of negative examples to include for each positive example. The function also shuffles the examples and scores while maintaining example, 
answer pairs. This is nice so that a training set can be used in order while maintaining an even mix of positive and negative examples accross batches.

```
    Parameters: pos_list: list of positive examples to include in ATCG encoding
                neg_list: list of negative examples to include in ATCG encoding
                neg_scale: int, how much larger the negative set will be than the positive set. 
    returns:    shuffle_train_list: list of ATCG encoded examples, contains positive and negative examples in random order
                shuffle_answer_list: list of correct classifications, in the same order as shuffle_Train_list
```