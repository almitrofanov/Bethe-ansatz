# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:51:25 2020

@author: Alexander Mitrofanov
"""

import numpy as np

def H_BlockDiag(hamiltonian_not_diag, n):
    H = hamiltonian_not_diag[0]
    basis = hamiltonian_not_diag[1]
    basis_diag = basis_diag_func(n, basis)
    I=np.zeros((2**n,2**n))
    for i in range(2**n):
        for j in range(2**n):
            bool_array = basis[i] == basis_diag[j]
            if np.all(bool_array):
                I[i,j] = 1
    H_diag=I.transpose().dot(H)
    H_diag=H_diag.dot(I)
    return H_diag, basis_diag
    
def basis_diag_func(n, basis): 
    fir_counter = 0  
    basis_diag=[]
    k = bin(2**n-1)
    base = np.zeros(n).astype(int)
    for i in range(len(k)-2):
            base[-i-1] = int(k[-i-1]) 
    basis_diag.append(base)                 

    for i in range(n):
        number_of_states=int(np.math.factorial(n)/(np.math.factorial(n-i-1)*np.math.factorial(i+1)))
        for j in range(number_of_states):
            fir_counter = fir_counter + 1
            sec_counter = 0
            number_of_zeros = -1     
            was_it_before = False
            while number_of_zeros != i+1 or was_it_before:
                k_initial = bin(2**n - 2**(i+1) - sec_counter)
                base = np.zeros(n).astype(int)
                for k in range(len(k_initial)-2):
                    base[-k-1] = int(k_initial[-k-1])
                if fir_counter + 1 > len(basis_diag):
                    basis_diag.append(base)
                else:
                     basis_diag[fir_counter] = base                   
                param = basis_diag[fir_counter]
                number_of_zeros = number_of_zeros_func(param)
                was_it_before = was_it_before_func(fir_counter, basis_diag, i+1)
                sec_counter = sec_counter + 1
    return basis_diag
                
def was_it_before_func(fir_counter, basis_diag, i):
    size_A =len(basis_diag)
    if 2**i -1 >= size_A-1:
        out = False
        return out
    start = 2**i-2 
    end = size_A-2
    for k in range(start, end+1):
        bool_array = basis_diag[k] == basis_diag[fir_counter]     
        if np.all(bool_array):
            out = True
            return out 
        else:
            out = False            
    return out 
                
def number_of_zeros_func(A):
    counter_zeros = 0
    A = str(A)
    size_A = len(A)
    for i in range(size_A):
        if A[i] == '0':
            counter_zeros = counter_zeros + 1
    return counter_zeros
    
    
    
            
            