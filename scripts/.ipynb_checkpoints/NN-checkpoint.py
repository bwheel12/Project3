##development note: general structure is good right now. But should be simplified to make generalization to any architecture easy and troubleshooting not the worst thing in the world

import numpy as np
import math as m
class NeuralNetwork:
    def __init__(self, setup, act_f,alpha,lamba,batch_size,bias):
        #example setup: [8,3,8], where first is input, last is output
        #provided dummy inputs: setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0
        #Note - these paramaters are examples, not the required init function parameters
        self.alpha = alpha
        self.lamba = lamba
        self.batch_size = batch_size
        self.bias = bias
        self.setup = setup
        self.edge_matrices = []
        self.input_layer   = []
        self.biases        = []
        self.layer_z = []
        self.layer_a = []
        self.number_layers = len(setup)-1 #the number of non-input layer, layers
        for i in range(self.number_layers):
            self.edge_matrices.append(np.random.normal(0,0.01,size=(setup[i+1],setup[i])))
            #self.edge_matrices.append(np.ones((setup[i+1],setup[i])))
            self.biases.append(np.random.normal(0,0.01,size=(setup[i+1])))
            #self.biases.append(np.zeros((setup[i+1])))
        
        for j in range(0,self.number_layers):
            self.layer_z.append(np.zeros((setup[j+1])))
            self.layer_a.append(np.zeros((setup[j+1])))
            
        self.act_f = act_f

    def get_single_input(self,input_layer):
        self.input_layer = input_layer
        
    def get_training_set(self,inputs):
        self.training_set = []
        for x in inputs:
            self.training_set.append(x)
    
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
        
        
        num_batch = len(batch_set)
        print("num batch is ",num_batch,end='\r')
        epoch_cost = 0
        for i in range(num_batch):
            #print("i is ",i,end='\r')
            self.get_single_input(batch_set[i])
            self.feedforward()
            partial_Ws, partial_bs = self.backprop(self.edge_matrices,self.biases,self.input_layer,self.layer_z,self.layer_a)
            for j in range(self.number_layers):
                
                delta_Ws[j] = delta_Ws[j] + partial_Ws[j]
                delta_bs[j] = delta_bs[j] + partial_bs[j]
                
        for z in range(self.number_layers):
            self.edge_matrices[z] = self.edge_matrices[z] - self.alpha*((1/(num_batch)*delta_Ws[z])+self.lamba*self.edge_matrices[z])
            #print(self.alpha*((1/(num_batch)*delta_Ws[z])+self.lamba*self.edge_matrices[z]))
            self.biases[z]        = self.biases[z]        - self.alpha*(1/(num_batch)*delta_bs[z])
            
        epoch_cost = 1/2*(self.input_layer-self.layer_a[self.number_layers-1])*(self.input_layer-self.layer_a[self.number_layers-1])
        #print("epoch cost is ",epoch_cost)
            
        return epoch_cost    
        
    def test_backprop(self):
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
            

    def fit(self):
        pass

    def predict(self):
        pass
    
    def sigmoid_deriv(self,y):
        return self.act_f(y)*(1-self.act_f(y))
        

def activation(x):
    """This function implements a sigmoid activation function
    parameters: x float
    returns:    f_of_x float bounded 0 to 1
    """
    #print("x is ",x,end='\r')
    f_of_x =  1/(1+m.exp(-x))
    return f_of_x
