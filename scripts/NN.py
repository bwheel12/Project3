##development note: general structure is good right now. But should be simplified to make generalization to any architecture easy and troubleshooting not the worst thing in the world
from scripts import io
import numpy as np
import math as m
class NeuralNetwork:
    def __init__(self, setup, act_f,alpha,lamba):
        #example setup: [8,3,8], where first is input, last is output
        #provided dummy inputs: setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0
        #Note - these paramaters are examples, not the required init function parameters
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
    """This function will provide k-fold cross validation on a given training set, with a given model in this case an untrained but instantiated nueral network.
    paramters: 
    
    """
    
    #first break inputs into K parts
    k_size = int(np.floor(len(train_set)/k))
    print("k size is",k_size)
    training_list = []
    answer_list = []
    for i in range(k):
        if i < k - 1:
            training_list.append(train_set[i*k_size:(i+1)*k_size])
            answer_list.append(answer_set[i*k_size:(i+1)*k_size])
        if i == k -1: 
            training_list.append(train_set[i*k_size:])
            answer_list.append(answer_set[i*k_size:])
        
    #print(list(training_list[0]))
    roc_FPRs = []
    roc_TPRs = []
    
    for j in range(k):
        temp_NN = NeuralNetwork(setup,activation,alpha,lamba)
        temp_train = []
        temp_answer = []
        temp_predict = []
        for m in range(k):
            if m != j:
                temp_train  = temp_train + list(training_list[m])
                temp_answer = temp_answer + list(answer_list[m])
            if m == j:
                temp_test = list(training_list[m])
                temp_test_answer = list(answer_list[m])
        temp_NN.train(epoch_number,batch_number,temp_train,temp_answer)
        num_test = len(temp_test)
        #predict each member of test set
        for z in range(num_test):
            temp_entry = io.one_hot_encode([temp_test[z]])
            temp_entry = temp_entry[0]
            temp_predict.append(temp_NN.predict(temp_entry)[0])
        #calculate FPR and TPR for various thresholds of outcomes
        
        print(temp_predict[0])
        print(temp_test_answer[0])
        temp_TPR = []
        temp_FPR = []
        for zed in range(10000):
            cutoff = zed*0.0001
            true_positive = 0
            true_negative = 0
            false_positive = 0
            false_negative = 0
            
            for n in range(len(temp_predict)):
                if temp_predict[n] > cutoff and np.isclose(temp_test_answer[n],1):
                    true_positive += 1
                if temp_predict[n] < cutoff and np.isclose(temp_test_answer[n],1):
                    false_negative += 1
                if temp_predict[n] > cutoff and np.isclose(temp_test_answer[n],0):
                    false_positive += 1
                if temp_predict[n] < cutoff and np.isclose(temp_test_answer[n],0):
                    true_negative += 1
            temp_TPR.append(true_positive/(true_positive+false_negative))
            temp_FPR.append(false_positive/(false_positive+true_negative))
    
        roc_FPRs.append(temp_FPR)
        roc_TPRs.append(temp_TPR)
        
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
    
    
    
    
    
def particle_swarm_optimization(train_set,answer_set,epoch_number,batch_number,setup,alpha,lamba,k):
    pass
    
    
    