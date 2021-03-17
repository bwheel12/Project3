from scripts import io
import numpy as np
import math as m
import random as rand

class NeuralNetwork:
    def __init__(self, setup, act_f,alpha,lamba):
        """This class implements a fully connected neural network. Setup should be a vector where the first entry indicates the input 
        layer size, the last the output layer size, and for any given number of entries in between the size of the respective hidden layers. 
        A function should be passed as act_f which is the activation function. alpha is the learning rate should be a scaler 0 to 1. 
        Lamba should be a very small scaler."""
        self.alpha = alpha #learning rate
        self.lamba = lamba #the weight decay constant
        self.setup = setup #the network architecture
        self.edge_matrices = [] 
        self.input_layer   = []
        self.biases        = []
        self.layer_z = [] #all the values of each layer after matrix multiplication, but before passing through the activation function
        self.layer_a = [] #all the values of each layer after passing through the activation function
        self.number_layers = len(setup)-1 #the number of non-input layer, layers
        #Initialize the edge and bias weights to small random numbers normaly distributed about 0
        for i in range(self.number_layers):
            self.edge_matrices.append(np.random.normal(0,0.01,size=(setup[i+1],setup[i])))
            self.biases.append(np.random.normal(0,0.01,size=(setup[i+1])))
        
        #initialize the layer nodes to the correct size for each layer
        for j in range(0,self.number_layers):
            self.layer_z.append(np.zeros((setup[j+1])))
            self.layer_a.append(np.zeros((setup[j+1])))
            
        self.act_f = act_f 

    def get_single_input(self,input_layer):
        """This function takes a single input and assigns it to the input layer
        Parameters: input layer, a vector the same length as the indicated input layer
        returns: none
        """
        self.input_layer = input_layer
        
    def get_training_set(self,inputs,answers):
        """This function takes a list of training examples and answers and stores them for training
        parameters: inputs, a list of one hot(or other wise numerically) encoded training examples
        answers: a list of answers to which the output layer should be compared. should be numerical and size match output layer"""
        self.training_set = []
        self.training_answers = []
        
        #check to make sure the inputs and answers are same length. if they arent perfectly matched then training is meaningless
        if len(inputs) != len(answers):
            raise ValueError('Training set does not match answers length')
        
        for x in inputs:
            self.training_set.append(x)
        
        for y in answers:
            self.training_answers.append(y)

    def feedforward(self):
        """This function does the matrix multiplication to translate the input layer into the output via passing through
        fully connected layers with the defined activation function. No parameters given acts entirely on self variables."""
        #first layer is unique as input layer has already been "activated"
        self.layer_z[0] = np.dot(self.edge_matrices[0],self.input_layer) + self.biases[0] #matrix mutliplication to creat z values
        self.layer_a[0] = self.act_vector(self.layer_z[0]) #a values come from doing the activation function on the z values
        
        #the rest of the layers easy and the same operations as the lines above just now the previous hidden layer is the input
        if self.number_layers > 1:
            for j in range(1,self.number_layers):
                self.layer_z[j] = np.dot(self.edge_matrices[j],self.layer_a[j-1]) + self.biases[j]
                self.layer_a[j] = self.act_vector(self.layer_z[j])
        
    def batch_descent(self):
        """This function uses the training inputs and answers to compare the current feedforward answer to the desired output.
        It then uses gradient descent to update the network weights to train the network. No parameters are given as it works 
        entirely on self variables"""
        #initialize the matrices that will hold the gradient values
        delta_Ws = []
        delta_bs = []
        
        for i in range(self.number_layers):
            delta_Ws.append(np.zeros((self.setup[i+1],self.setup[i])))
            delta_bs.append(np.zeros((self.setup[i+1])))
        
        
        #get the exmaples and answers to train on
        batch_set = self.training_set
        batch_ans = self.training_answers
        
        
        #multiple examples will be calculated and the gradients will be averaged before updating the network
        num_batch = len(batch_set)
        epoch_cost = 0
        #go through each examples one by one add up the gradients
        for i in range(num_batch):
            self.get_single_input(batch_set[i]) #assign to input
            self.feedforward() #calculate output
            partial_Ws, partial_bs = self.backprop(self.edge_matrices,self.biases,batch_ans[i],self.layer_z,self.layer_a) #backprop to calculate gradients for this example
            #for each layer add up the gradients as they are calculated one by one
            for j in range(self.number_layers):
                
                delta_Ws[j] = delta_Ws[j] + partial_Ws[j]
                delta_bs[j] = delta_bs[j] + partial_bs[j]
            epoch_cost = epoch_cost + 1/2*(batch_ans[i]-self.layer_a[self.number_layers-1])*(batch_ans[i]-self.layer_a[self.number_layers-1]) #add up the whole cost for this batch
        
        #update the weights and biases for each layer
        for z in range(self.number_layers):
            self.edge_matrices[z] = self.edge_matrices[z] - self.alpha*((1/(num_batch)*delta_Ws[z])+self.lamba*self.edge_matrices[z])
            self.biases[z]        = self.biases[z]        - self.alpha*(1/(num_batch)*delta_bs[z])
        
        #calculate the average cost for the batch. Epoch isnt exactly right in how I ended up using this. But not a good idea to change at this point
        epoch_cost = epoch_cost/num_batch
        
            
        return epoch_cost    
        
    def test_backprop(self):
        """This function only does the gradient calculation step within back prop. really useful for debugging. Kept altough no longer needed in final implementation"""
        #this function is primarily for debugging, particularly the autoencoder
        partial_Ws, partial_bs = self.backprop(self.edge_matrices,self.biases,self.input_layer,self.layer_z,self.layer_a)
        print(partial_Ws)
        print(partial_bs)
        
        
   
    def backprop(self,edge_matrices,biases,correct,layer_zs,layer_as):
        """This function does the gradient calculations. Might not fully encompas what might be considered 'backprop' but the essential difficult step. And
        it is too late to change the name now. It takes all the essential components of the network as they typically are. It returns the partial gradient values
        for a given correct answer."""
        
        last_layer_index = self.number_layers - 1 #the index of the final item in the layer or edge matrices list
                
         
        errors = [] #will hold the small d error for a layer
        #find the error for each node in each layer
        for j in range(last_layer_index,-1,-1):
            temp_errors = []
            temp_derivatives = self.derivatives_vector(layer_zs[j])
            if j == last_layer_index:
                temp_errors = -(correct - layer_as[j])*temp_derivatives
            else:
                temp_errors = (np.dot(np.transpose(edge_matrices[j+1]),errors[0]))*temp_derivatives
            errors.insert(0,temp_errors)
        
        self.errors = errors #this i think was there to trouble shoot, probably can be taken out. But dont want to break anything
        #just compute partial derivatives now
        partial_Ws = []
        partial_bs = []
        
        #partial derivates are the cross product of the erros above and the activations for W and just the errors for b. Edge cases have to be handled slightly specially
        for k in range(self.number_layers):
            if k ==0:
                partial_Ws.append(np.outer(errors[k],self.input_layer))
                partial_bs.append(errors[k])
            if k > 0:
                partial_Ws.append(np.outer(errors[k],layer_as[k-1]))
                partial_bs.append(errors[k])
                
        return partial_Ws, partial_bs
            
       
    def derivatives_vector(self,x):
        """This function is useful for backprop above. It takes a vector of numerical values and calculates the sigmoidal derivative
        value of each and returns it in a similar vector."""
        deriv_vect = []
        for i in range(len(x)):
            deriv_vect.append(self.sigmoid_deriv(x[i]))
        deriv_vect = np.array(deriv_vect)
        return deriv_vect
    
    def act_vector(self,x):
        """This function is useful for feedforward. It takes a vector of numerical values and calculates the activation function value 
        of each element and returns it in a similar vector"""
        act_vect = []
        for i in range(len(x)):
            act_vect.append(self.act_f(x[i]))
        act_vect = np.array(act_vect)
        return act_vect
            

    def train(self,epoch_number,batch_size, whole_set, answer_set):
        """This function trains the network by interating through a training set containing negative and positive examples and a corresponding answer set
        this function expects ATCG encoding as it does the one hot encoding call. This is also where batch size gets used to break the training set up into
        smaller chunks with which the weights get updated. I.e. at the end of the k loop the weights are actually updated"""
        
        cost_list = []
        
        for j in range(epoch_number):
    
            batch_cycles = int(np.ceil(len(whole_set)/batch_size)) #the number of batches in data set for a given batch size
            for k in range(batch_cycles):
                if k < batch_cycles-1:
                    training_list = whole_set[(k)*batch_size:(k+1)*batch_size]
                    training_list = io.one_hot_encode(training_list)
                    answer_list    = answer_set[(k)*batch_size:(k+1)*batch_size]
                if k == batch_cycles-1:
                    training_list = whole_set[(k)*batch_size:]
                    training_list = io.one_hot_encode(training_list)
                    answer_list   = answer_set[(k)*batch_size:]

                self.get_training_set(training_list,answer_list)
                cost_list.append(self.batch_descent())
        
        
        
        
        return cost_list
        
        
        
    def predict(self,input_layer):
        """This function takes a single input and gets the network prediction by feeding it forward through. The prediction is returned."""
        self.input_layer = input_layer
        self.feedforward()
        answer = self.layer_a[self.number_layers-1]
        return answer
        
    
    def sigmoid_deriv(self,y):
        "This function is the derivative of the sigmoidal activation function"
        return self.act_f(y)*(1-self.act_f(y))
        

def activation(x):
    """This function implements a sigmoid activation function
    parameters: x float
    returns:    f_of_x float bounded 0 to 1
    """
    f_of_x =  1/(1+m.exp(-x))
    return f_of_x


def k_fold_validation(train_set,answer_set,epoch_number,batch_number,setup,alpha,lamba,k):
    """This function will provide k-fold cross validation on a given training set, with a given model. In this case the parameters needed create the NN are passed
    and each validation loop will create a new temporary network with which to train and validate. For each temporarily model it will evaluate on the held out validation
    set and calculate True Positive Rates and False Positive Rates at a series of threshold values between 0 and 1. It returns these True Positive Rates and False Positive 
    rates as nested lists with a list within each for each validation cycle.
    
    """
    
    #first break inputs into K parts
    k_size = int(np.floor(len(train_set)/k))
    print("k size is",k_size,end='\r')
    training_list = []
    answer_list = []
    for i in range(k):
        #the first k-1 parts are equal size
        if i < k - 1:
            training_list.append(train_set[i*k_size:(i+1)*k_size-1])
            answer_list.append(answer_set[i*k_size:(i+1)*k_size-1])
        #the last part is from the starting index to the end and will be shorter than the other parts
        if i == k -1: 
            training_list.append(train_set[i*k_size:])
            answer_list.append(answer_set[i*k_size:])
        
    #print(list(training_list[0]))
    #this part is to create nested lists of True Positive Rate and False positive Rates for predictions on the out set for differenent 0 to 1 thresholds
    roc_FPRs = [] #nested list of False positive rates
    roc_TPRs = [] #nested list of true positive rates
    
    #need to iterate large loop for each k validation
    for j in range(k):
        temp_NN = NeuralNetwork(setup,activation,alpha,lamba) # initialize a temporary NN to train
        temp_train = [] #will be the training examples for this validation
        temp_answer = [] #will be the training answers for this validation
        temp_predict = [] #will be the predictios for this validation
        for m in range(k):
            #use the Kth set from above to be the validation set
            if m != j:
                temp_train  = temp_train + list(training_list[m]) #string together all other sets to be training inputs
                temp_answer = temp_answer + list(answer_list[m]) #string together all other sets to be the trainig aswers
            if m == j:
                temp_test = list(training_list[m]) #the Kth set is the validation test inputs
                temp_test_answer = list(answer_list[m]) #the Kth answer set is the validation answer set
        temp_NN.train(epoch_number,batch_number,temp_train,temp_answer) #train the temporary NN with the given parameters and subdivided training set from immediately above lines
        num_test = len(temp_test) #number of validation set examples
        #predict each member of test set
        for z in range(num_test):
            temp_entry = io.one_hot_encode([temp_test[z]]) #one hot encode the given example, even though it is one entry it must be given as a list
            temp_entry = temp_entry[0] #above returns a list, but predict wants only one example
            temp_predict.append(temp_NN.predict(temp_entry)[0]) #get the predicted value
        
        #calculate FPR and TPR for various thresholds of outcomes
        #print(temp_predict[0])
        #print(temp_test_answer[0])
        temp_TPR = [] #temporary list to hold  true positive rates for this validation iteration
        temp_FPR = [] #temporary list to hold false positive rates for this validation iteration
        for zed in range(10):
            cutoff = zed*0.1 #hard coded that predictions are between 0 to 1 and stepping through the predictions with a threshold step of .1
            true_positive = 0 #temp counting variable
            true_negative = 0 #temp counting variable
            false_positive = 0 #temp counting variable
            false_negative = 0 #temp counting variable
            
            #count up the results for the given threshold value of the loop
            for n in range(len(temp_predict)):
                if temp_predict[n] > cutoff and np.isclose(temp_test_answer[n],1):
                    true_positive += 1
                if temp_predict[n] < cutoff and np.isclose(temp_test_answer[n],1):
                    false_negative += 1
                if temp_predict[n] > cutoff and np.isclose(temp_test_answer[n],0):
                    false_positive += 1
                if temp_predict[n] < cutoff and np.isclose(temp_test_answer[n],0):
                    true_negative += 1
            #use the temporary counting variables to determine the TPR and FPR and add it to temp list
            temp_TPR.append(true_positive/(true_positive+false_negative))
            temp_FPR.append(false_positive/(false_positive+true_negative))
    
        roc_FPRs.append(temp_FPR) #add the temporarily list to a new slot in the nested list to return
        roc_TPRs.append(temp_TPR) #add the temporarily list to a new slot in the nested list to return
        
    return roc_FPRs, roc_TPRs
            
    
    

    
def roc_auc_list(roc_TPRs,roc_FPRs):
    """This function expects lists of lists of TPRs and FPRs (ie at least 2 sets per TPR and FPR). With those lists it calculates the area under the 
    curve of each corresponding ROC curve."""
    roc_AUCs =  [] #outermost list
    num_iter = len(roc_TPRs) #number of additions to do
    
    #sum up the area using the trapezoidal rule
    for i in range(num_iter):
        TPR_len = len(roc_TPRs[i])
        AUC_temp = 0
        for j in range(1,TPR_len):
            AUC_temp += (roc_FPRs[i][j-1]-roc_FPRs[i][j])*(1/2)*(roc_TPRs[i][j-1]+roc_TPRs[i][j])
            
        roc_AUCs.append(AUC_temp)
        
        
    return roc_AUCs
    
    
    
    
    
def genetic_optimization(train_set,answer_set,generation_num,pop_num,cross_rate,mutation_rate, centers):
    """This function implements a genetic algorithm to optimize the network, this assumes a network with a 68 node input layer,
    a 1 node output layer, and 2  hidden layers. I will seek to optmize the size of each hidden layer, the learning rate, the rate of weight decay, 
    the number of epochs to perform, and the batch size. I will use the average auROC from k-fold cross validation to evaluate each population member.
    pop_num should be even. 
    """
    
    #centers order hidden1, hidden2, alpha, lamba, epoch, batch size
    #hidden1 [1,inf]
    #hidden2 [1,inf]
    #alpha   [0,1]
    #lamba   [0,1]
    #epoch   [1,inf]
    #batch size [1,len(train_set)]
    centers = centers #these values will be the population center with which generation 0 will be created
    generations = []
    train_len =  len(train_set)
    optimize_num = len(centers)
    #create generation 0
    for i in range(pop_num):
        temp_individual = []
        for j in range(optimize_num):
            temp_individual.append(np.random.normal(centers[j],centers[j]*0.5))
        generations.append(check_individual(temp_individual,train_len))
    
    generations = [generations]
    gen_scores = [score_generation(generations[0],train_set,answer_set)]
    gen_avg_score = [np.mean(gen_scores)]
    gen_max_score = [np.max(gen_scores)]
    
    #conduct evolution
    trnment_num = int(pop_num/2)
    for k in range(1,generation_num):
        temp_gen = []
        for m in range(trnment_num):
            #generate 4 individuals for mini tournament
            temp_index_1 = int(np.floor(rand.random()*pop_num))
            temp_index_2 = int(np.floor(rand.random()*(pop_num-1.0001)))
            temp_index_3 = int(np.floor(rand.random()*pop_num))
            temp_index_4 = int(np.floor(rand.random()*(pop_num-1.0001)))
            if temp_index_1 == temp_index_2:
                temp_index_2 += 1
            if temp_index_3 == temp_index_4:
                temp_index_4 += 1
            
            temp_individ_1 = generations[k-1][temp_index_1]
            temp_individ_2 = generations[k-1][temp_index_2]
            temp_individ_3 = generations[k-1][temp_index_3]
            temp_individ_4 = generations[k-1][temp_index_4]
            
            #generate parents from tournament, arbitrarily prefer 1>2 and 3>4
            if gen_scores[k-1][temp_index_1] >= gen_scores[k-1][temp_index_2]:
                parent_1 = temp_individ_1
            if gen_scores[k-1][temp_index_1] < gen_scores[k-1][temp_index_2]:
                parent_1 = temp_individ_2
            if gen_scores[k-1][temp_index_3] >= gen_scores[k-1][temp_index_4]:
                parent_2 = temp_individ_3
            if gen_scores[k-1][temp_index_3] < gen_scores[k-1][temp_index_4]:
                parent_2 = temp_individ_4
            
            #conduct cross over
            
            for z in range(optimize_num):
                if rand.random() < cross_rate:
                    temp_value = parent_1[z]
                    parent_1[z] = parent_2[z]
                    parent_2[z] = temp_value
                    
            #conduct mutation for parent1
            for zed in range(optimize_num):
                if rand.random() < mutation_rate:
                    if rand.random() > 0.5:
                        parent_1[zed] = parent_1[zed]*1.15
                    else:
                        parent_1[zed] = parent_1[zed]*0.85
                        
            #conduct mutation for parent2
            for zed in range(optimize_num):
                if rand.random() < mutation_rate:
                    if rand.random() > 0.5:
                        parent_1[zed] = parent_2[zed]*1.15
                    else:
                        parent_1[zed] = parent_2[zed]*0.85
    
            child_1 = check_individual(parent_1,train_len)
            child_2 = check_individual(parent_2,train_len)
            
            temp_gen.append(child_1)
            temp_gen.append(child_2)
        print(k,"evolutions have been performed")
        
        generations.append(temp_gen)
        gen_scores.append(score_generation(temp_gen,train_set,answer_set))
        gen_avg_score.append(np.mean(gen_scores[k]))
        gen_max_score.append(np.max(gen_scores[k]))

    print("The generation scores are",gen_scores)
    best_individual = generations[generation_num-1][np.argmax(gen_scores[generation_num-1])]
    
    return best_individual, gen_avg_score, gen_max_score, generations
    
    

def check_individual(individual,train_len):
    """This function is needed to keep the parameters manipulated by the genetic algorithm above within acceptable contexts. Accepts a population
    individual in the paradigm as above and train_len which is the length of the training set. This is important because it is the limit for 
    batch size. It generally fixes errors beyond a bound by reseting it to an acceptable boundary condition"""
    individual[0] = int(np.floor(individual[0])) #hidden layer 1 should be an integer
    individual[1] = int(np.floor(individual[1])) #hidden layer 2 should be an integer
    individual[4] = int(np.floor(individual[4])) #epoch number should be an integer
    individual[5] = int(np.floor(individual[5])) #batch size should be an integer
    
    #hidden layer must be between 1 and inf
    if individual[0] < 1:
        individual[0] = 1
    #hidden layer must be between 1 and inf
    if individual[1] < 1:
        individual[1] = 1
    #alpha must be between 0 and 1
    if individual[2] >= 1:
        individual[2] = 0.999
    if individual[2] <= 0:
        individual[2] = 0.001
    #lamba must be between 0 and 1
    if individual[3] >= 1:
        individual[3] = 0.999
    if individual[3] <= 0:
        individual[3] = 0
    #epoch number must be between 1 and inf
    if individual[4] < 1:
        individual[4] = 1
    #batch size must be between 1 and the length of the training set
    if individual[5] < 1:
        individual[5] = 1
    if individual[5] > train_len:
        individual[5] = train_len
        
    return individual
    
def score_generation(generation,train_set,answer_set):
    """This function scores a generation from the genetic algorithm by calculating the auROC for the given parameters of an individual.
    It is hard coded for k=2 cross validation to save time in scoring generations. Essentially it is the standard initial,train,evaluate for
    k-fold cross validation just looped over many temporary networks. Requires a list of 'individuals' and the training and answer set. The training set should be
    in ATCG encoding. It returns a vector of the scores in the order of the generation individuals."""
    generation_scores = []
    
    for ind in generation:
        #instantiate the neural network
        setup = [68,ind[0],ind[1],1]
        alpha = ind[2]
        lamba = ind[3] #0.1 is good
        batch_size = ind[5]
        epoch_number = ind[4]
        k = 2
        #run the validation
        roc_FPRs, roc_TPRs  = k_fold_validation(train_set,answer_set,epoch_number,batch_size,setup,alpha,lamba,k)
        #get the auROC as the score
        roc_AUCs = roc_auc_list(roc_TPRs,roc_FPRs)
        generation_scores.append(np.mean(roc_AUCs))
        
    return generation_scores