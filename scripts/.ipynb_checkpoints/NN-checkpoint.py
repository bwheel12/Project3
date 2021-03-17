##development note: general structure is good right now. But should be simplified to make generalization to any architecture easy and troubleshooting not the worst thing in the world
from scripts import io
import numpy as np
import math as m
import random as rand

class NeuralNetwork:
    def __init__(self, setup, act_f,alpha,lamba):
        self.alpha = alpha
        self.lamba = lamba
        self.setup = setup
        self.edge_matrices = []
        self.input_layer   = []
        self.biases        = []
        self.layer_z = []
        self.layer_a = []
        self.number_layers = len(setup)-1 #the number of non-input layer, layers
        for i in range(self.number_layers):
            self.edge_matrices.append(np.random.normal(0,0.01,size=(setup[i+1],setup[i])))
            self.biases.append(np.random.normal(0,0.01,size=(setup[i+1])))
        
        for j in range(0,self.number_layers):
            self.layer_z.append(np.zeros((setup[j+1])))
            self.layer_a.append(np.zeros((setup[j+1])))
            
        self.act_f = act_f

    def get_single_input(self,input_layer):
        self.input_layer = input_layer
        
    def get_training_set(self,inputs,answers):
        self.training_set = []
        self.training_answers = []
        
        if len(inputs) != len(answers):
            raise ValueError('Training set does not match answers length')
        
        for x in inputs:
            self.training_set.append(x)
        
        for y in answers:
            self.training_answers.append(y)
    
    def make_weights(self):
        pass

    def feedforward(self):
        self.layer_z[0] = np.dot(self.edge_matrices[0],self.input_layer) + self.biases[0]
        self.layer_a[0] = self.act_vector(self.layer_z[0])
        
        if self.number_layers > 1:
            for j in range(1,self.number_layers):
                #print(j)
                self.layer_z[j] = np.dot(self.edge_matrices[j],self.layer_a[j-1]) + self.biases[j]
                self.layer_a[j] = self.act_vector(self.layer_z[j])
        
    def batch_descent(self):
        delta_Ws = []
        delta_bs = []
        
        for i in range(self.number_layers):
            delta_Ws.append(np.zeros((self.setup[i+1],self.setup[i])))
            delta_bs.append(np.zeros((self.setup[i+1])))
        
        
        #need to determine how a batch is set up
        #for testing use whole training set
        batch_set = self.training_set
        batch_ans = self.training_answers
        
        
        num_batch = len(batch_set)
        #print("num batch is ",num_batch,end='\r')
        epoch_cost = 0
        for i in range(num_batch):
            #print("i is ",i,end='\r')
            self.get_single_input(batch_set[i])
            #print("input length is",len(batch_set[i]))
            self.feedforward()
            partial_Ws, partial_bs = self.backprop(self.edge_matrices,self.biases,batch_ans[i],self.layer_z,self.layer_a)
            for j in range(self.number_layers):
                
                delta_Ws[j] = delta_Ws[j] + partial_Ws[j]
                delta_bs[j] = delta_bs[j] + partial_bs[j]
            epoch_cost = epoch_cost + 1/2*(batch_ans[i]-self.layer_a[self.number_layers-1])*(batch_ans[i]-self.layer_a[self.number_layers-1])
                
        for z in range(self.number_layers):
            self.edge_matrices[z] = self.edge_matrices[z] - self.alpha*((1/(num_batch)*delta_Ws[z])+self.lamba*self.edge_matrices[z])
            #print(self.alpha*((1/(num_batch)*delta_Ws[z])+self.lamba*self.edge_matrices[z]))
            self.biases[z]        = self.biases[z]        - self.alpha*(1/(num_batch)*delta_bs[z])
        
        
        epoch_cost = epoch_cost/num_batch
        
            
        return epoch_cost    
        
    def test_backprop(self):
        #this function is primarily for debugging, particularly the autoencoder
        partial_Ws, partial_bs = self.backprop(self.edge_matrices,self.biases,self.input_layer,self.layer_z,self.layer_a)
        print(partial_Ws)
        print(partial_bs)
        
        
   
    def backprop(self,edge_matrices,biases,correct,layer_zs,layer_as):
        last_layer_index = self.number_layers - 1 #the index of the final item in the layer or edge matrices list
                
        ##something is wrong here...    
        errors = [] 
        for j in range(last_layer_index,-1,-1):
            temp_errors = []
            temp_derivatives = self.derivatives_vector(layer_zs[j])
            if j == last_layer_index:
                temp_errors = -(correct - layer_as[j])*temp_derivatives
            else:
                #print(last_layer_index-j-1)
                #print(edge_matrices[last_layer_index-j-1])
                temp_errors = (np.dot(np.transpose(edge_matrices[j+1]),errors[0]))*temp_derivatives
            errors.insert(0,temp_errors)
        
        self.errors = errors
        #print(len(errors))
        #print("errors are ",errors)
        #just compute partial derivatives now
        partial_Ws = []
        partial_bs = []
        
        
        for k in range(self.number_layers):
            if k ==0:
                partial_Ws.append(np.outer(errors[k],self.input_layer))
                partial_bs.append(errors[k])
            if k > 0:
                partial_Ws.append(np.outer(errors[k],layer_as[k-1]))
                partial_bs.append(errors[k])
            
            #partial_Ws.insert(0,np.outer(errors[last_layer_index-k],layer_as[last_layer_index-k-1]))
            #partial_bs.insert(0,errors[last_layer_index-k])
                
        return partial_Ws, partial_bs
            
       
    def derivatives_vector(self,x):
        deriv_vect = []
        for i in range(len(x)):
            deriv_vect.append(self.sigmoid_deriv(x[i]))
        deriv_vect = np.array(deriv_vect)
        return deriv_vect
    
    def act_vector(self,x):
        act_vect = []
        for i in range(len(x)):
            act_vect.append(self.act_f(x[i]))
        act_vect = np.array(act_vect)
        return act_vect
            

    def train(self,epoch_number,batch_size, whole_set, answer_set):
        """This function trains the network by interating through a training set containing negative and positive examples and a corresponding answer set
        this function expects ATCG encoding"""
        
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
    and each validation loop will create a new temporary 
    
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
            cutoff = zed*0.1 #hard coded that predictions are between 0 to 1 and stepping through the predictions with a threshold step of .0001
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
    """This function expects lists of lists of TPRs and FPRs (ie at least 2 sets per TPR and FPR)"""
    roc_AUCs =  []
    num_iter = len(roc_TPRs)
    
    for i in range(num_iter):
        TPR_len = len(roc_TPRs[i])
        AUC_temp = 0
        for j in range(1,TPR_len):
            AUC_temp += (roc_FPRs[i][j-1]-roc_FPRs[i][j])*(1/2)*(roc_TPRs[i][j-1]+roc_TPRs[i][j])
            
        roc_AUCs.append(AUC_temp)
        
        
    return roc_AUCs
    
    
    
    
    
def genetic_optimization(train_set,answer_set,generation_num,pop_num,cross_rate,mutation_rate, centers):
    #should keep track of average fitness, and best fitness
    """This function implements a genetic algorithm to optimize the network, this assumes a network with a 68 node input layer,
    a 1 node output layer, and 2  hidden layers. I will seek to optmize the size of each hidden layer, the learning rate, the rate of weight decay, 
    the number of epochs to perform, and the batch size. I will use the average auROC from k-fold cross validation to evaluate each pop member
    pop_num should be even
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
    individual[0] = int(np.floor(individual[0]))
    individual[1] = int(np.floor(individual[1]))
    individual[4] = int(np.floor(individual[4]))
    individual[5] = int(np.floor(individual[5]))
    
    if individual[0] < 1:
        individual[0] = 1
    if individual[1] < 1:
        individual[1] = 1
    if individual[2] >= 1:
        individual[2] = 0.999
    if individual[2] <= 0:
        individual[2] = 0.001
    if individual[3] >= 1:
        individual[3] = 0.999
    if individual[3] <= 0:
        individual[3] = 0
    if individual[4] < 1:
        individual[4] = 1
    if individual[5] < 1:
        individual[5] = 1
    if individual[5] > train_len:
        individual[5] = train_len
        
    return individual
    
def score_generation(generation,train_set,answer_set):
    generation_scores = []
    
    for ind in generation:
        #instantiate the neural network
        setup = [68,ind[0],ind[1],1]
        alpha = ind[2]
        lamba = ind[3] #0.1 is good
        batch_size = ind[5]
        epoch_number = ind[4]
        k = 2
        roc_FPRs, roc_TPRs  = k_fold_validation(train_set,answer_set,epoch_number,batch_size,setup,alpha,lamba,k)
        roc_AUCs = roc_auc_list(roc_TPRs,roc_FPRs)
        generation_scores.append(np.mean(roc_AUCs))
        
    return generation_scores