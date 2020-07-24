# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:42:23 2020

@author: Alexander Mitrofanov
"""

import numpy as np
import math
from HeisenbergHamiltonian import Ham_xxz
from Helpers import *
from BlockDiagonalization import H_BlockDiag
from Bethe_FM import *

## constants
n = 6
J = 1
J_a=np.array([1, 1, 1]) #exchange anisotropy
border = 'close'

## Hamiltonian
hamiltonian_not_diag = Ham_xxz(J, J_a, n, border)
hamiltonian_diag = H_BlockDiag(hamiltonian_not_diag, n)
[H_diag, basis_diag] = H_BlockDiag(hamiltonian_not_diag, n)

del hamiltonian_diag, hamiltonian_not_diag
eig = np.linalg.eig(H_diag) #eigen vectors and eigen values
eig_val = clean(eig[0], 1e-10)
eig_vec = clean(eig[1], 1e-10)
del eig
H_1magnon = H_diag[1:n+1, 1:n+1]
length_2 = int(n*(n-1)/2)
H_2magnon = H_diag[n+1:n+1+length_2, n+1:n+1+length_2]

H_3magnon = H_diag[22:42,22:42]    
eig = np.linalg.eig(H_3magnon) #eigen vectors and eigen values
eig_val3 = clean(eig[0], 1e-10)
eig_vec3 = clean(eig[1], 1e-10)

## Bethe ansatz
[states_one_magnon, E_1magnon] = one_magnon(J, n)
[a_1, a_2, a_3, E_1, E_2, E_3] = two_magnons(J, n)

#forming hamiltonian

# checking solutions
print('\n 1 magnon \n')  
for i in range(n): # checking
    IfEigen(H_1magnon, E_1magnon[i], states_one_magnon[:, i])
print('\n 2 magnons \n')
print('C1 class \n')      
for i in range(n): 
    IfEigen(H_2magnon, E_1[i], a_1[:, i]) # checking
print('\n C2 class \n')  
for i in range(int(n*(n-5)/2 + 3)):
    IfEigen(H_2magnon, E_2[i], a_2[:, i])  # checking
print('\n C3 class \n')
if n > 4:      
    for i in range(int(n-3)):
        IfEigen(H_2magnon, E_3[i], a_3[:, i])  # checking
    
# forming states in full basis
H_bethe = FormingHamiltonian(n, states_one_magnon, a_1, a_2, a_3, eig_vec3)


# Total angular momentum for each state
s_s_1=[]
for i in range(42):
    s_s_1.append(FullAngMom(H_bethe[:,i], n))
# s_s_1=[]
# for i in range(2**n):
#     s_s_1.append(FullAngMom(eig_vec[:,i], n))



    
    







