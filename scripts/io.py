import numpy as np
import random as rand

def import_positives(import_path):
    """This function expects the path to a file to import DNA sequences for the positve examples and provides a list 
    where each entry is a string from a line in the origianl file"""
    positive_list = []
    with open(import_path) as f:
        lines = f.readlines()
    
    for x in lines:
        positive_list.append(x[0:17])
    
    
    return positive_list


def import_negatives(import_path,k_mer):
    """This function takes the path to the 1K length fasta upstream sequences
    it randomly selects a k_mer length string from a random sequence line from each gene"""
    
    negative_list = []
    
    with open(import_path) as f:
        lines = f.readlines()
    
    num_lines = len(lines)
    
    for i in range(num_lines):
        if lines[i][0] == '>':
            temp_gene = lines[(i):(i+1)*18-1]
            rand_line = int(np.ceil(rand.random()*16))
            rand_index = int(np.floor(rand.random()*(61-17)))
            negative_list.append(temp_gene[rand_line][rand_index:(rand_index+k_mer)])
        
        
    
    
    
    return negative_list


def one_hot_encode(k_mer_list):
    """This function takes a list of k-mers A,G,C,T and returns a list where each entry is a 1-hot encoded version of the k-mer"""
    one_hot_encoded = []
    num_kmers = len(k_mer_list)
    for i in range(num_kmers):
        temp_kmer = k_mer_list[i]
        len_kmer = len(temp_kmer)
        temp_1_hot = np.zeros((4*len_kmer))
        for j in range(len_kmer):
            if temp_kmer[j] == 'A':
                sub_index = 0
            if temp_kmer[j] == 'T':
                sub_index = 1
            if temp_kmer[j] == 'G':
                sub_index = 2
            if temp_kmer[j] == 'C':
                sub_index = 3
            temp_1_hot[(j-1)*4+sub_index] = 1
        one_hot_encoded.append(temp_1_hot)
    
    
    return one_hot_encoded