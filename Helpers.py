# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:51:28 2020

@author: Alexander Mitrofanov
"""

import numpy as np
import math
from BlockDiagonalization import H_BlockDiag
from HeisenbergHamiltonian import Ham_xxz
from scipy.optimize import fsolve
from scipy.optimize import newton_krylov
from scipy.optimize import anderson

def clean(A, order):
    # A =np.matrix(A)
    # A= complex(A)
    if len(A.shape) > 1:
        length = A.shape[0]
        width = A.shape[1]
        for j in range(width):
            for i in range(length):
                if abs(A[i,j].real) < order:
                    A[i,j] = 1j*A[i,j].imag
                if abs(A[i,j].imag) < order:
                    A[i,j] = A[i,j].real
        return A    
    else:
        length = A.shape[0]
        for i in range(length):           
            if abs(A[i].real) < order:
                A[i] = 1j*A[i].imag
            if abs(A[i].imag) < order:
                A[i] = A[i].real
        return A

def IfEigen(H, E, psi):
    # IfEigen(H_diag, eig_val[0], eig_vec[:, 0])
    PsiE1 = H.dot(psi) # matrix multiplication of Matrix H and vector psi
    PsiE2 = E*psi # normalyzed eigen states
    PsiE1 = np.round(PsiE1, 3)
    PsiE2 = np.round(PsiE2, 3)
    sum_elem = 0  # sum of elements
    for i in range(PsiE1.shape[0]):
        if PsiE1[i] == PsiE2[i]:
            sum_elem = sum_elem + 1
    if sum_elem == PsiE1.shape[0]:
        print('eigen')
    else:
         print('not eigen')
         
         
def normalyze(a):
    norm = 0
    if len(a.shape) > 1:
        width = a.shape[0]
        length = a.shape[1]
        a_normalyzed = np.zeros([width, length])
        a_normalyzed  = a_normalyzed.astype(complex)
        for j in range(length):
            for i in range(width):
                norm = norm + abs(a[i,j])**2
            a_normalyzed[:,j] = (1/math.sqrt(norm)) * a[:, j]
            norm = 0
        return a_normalyzed
    else:
        width = a.shape[0]
        a_normalyzed = np.zeros([width, 1])
        a_normalyzed  = a_normalyzed.astype(complex)
        for i in range(width):
            norm = norm + abs(a[i])**2
        a_normalyzed = (1/math.sqrt(norm))*a
        return a_normalyzed

def solving_C2(k_0, lambd, n):    
    func = lambda k_1 : (2*((np.tan(n*k_1/2))**-1) - (((np.tan(k_1/2))**-1)-((np.tan((k_0-k_1)/2)))**-1))
    k_initial_guess = 3
    # k_1 = fsolve(func, k_initial_guess, xtol=1e-12, maxfev = 10000)
    k_1 = newton_krylov(func, k_initial_guess, f_tol=1e-12)
    # k_1 = anderson(func, k_initial_guess, iter = 10000, f_tol = 1e-12)
    k_2 = k_0 - k_1
    theta = 2*math.pi*lambd[1]-n*k_2
    return k_1, k_2, theta

def DeltaFunction(a, b):
    if a == b:
        result = 1
    else:
        result = 0
    return result

# def solving_C3():
    

def ChangeOrder(matrix):
    length = matrix.shape[0]
    width = matrix.shape[1]
    matrix_changed = np.zeros([length, width])
    matrix_changed  = matrix_changed.astype(complex)
    for i in range(length):
        matrix_changed[i, :] = matrix[length-i-1, :]
    return matrix_changed

def solving_C3(k_0, lambd, n):
    phi = np.pi*(lambd[0]-lambd[1])
    res = 0
    for l in range(10):
        if res == 0:
            func = lambda nu : ((np.cos(k_0/2))*np.sinh(n*nu) - np.sinh((n-1)*nu) - np.cos(phi)*np.sinh(nu))
            nu_initial_guess = 3
            nu = fsolve(func, nu_initial_guess, xtol=1e-12, maxfev = 10000)
            # nu = newton_krylov(func, nu_initial_guess, f_tol=1e-12)
            # nu = anderson(func, nu_initial_guess, iter = 10000, f_tol = 1e-12)
            res = nu
    if abs(nu) > 0:
        k_1 = k_0/2 + 1j*nu
        k_2 = k_0/2 - 1j*nu
        theta = phi + 1j*n*nu
    else:
        k_1 = np.inf
        k_2 = np.inf
        theta = np.inf
    return k_1, k_2, theta
            
            
def FullAngMom(state, n):
    state_line = np.matrix(state)
    state_column = np.transpose(state_line)
    state_line = np.conj(state_line)
    Sx =0.5*np.array([[0, 1], [1, 0]])
    Sy =0.5*np.array([[0, -1j], [1j, 0]])
    Sz =0.5*np.array([[1, 0], [0, -1]])
    Sx_total = np.zeros([2**n, 2**n])
    Sy_total = np.zeros([2**n, 2**n])
    Sz_total = np.zeros([2**n, 2**n])
    Sx_total  = np.kron(np.identity(2**(n-1)), Sx) + np.kron(Sx, np.identity(2**(n-1)))
    Sy_total  = np.kron(np.identity(2**(n-1)), Sy) + np.kron(Sy, np.identity(2**(n-1))) 
    Sz_total  = np.kron(np.identity(2**(n-1)), Sz) + np.kron(Sz, np.identity(2**(n-1)))      
    for i in range(1, n-1):           
        Sx_total = Sx_total + np.kron(np.kron(np.identity(2**i), Sx), np.identity(2**(n-1-i))) 
        Sy_total = Sy_total + np.kron(np.kron(np.identity(2**i), Sy), np.identity(2**(n-1-i))) 
        Sz_total = Sz_total + np.kron(np.kron(np.identity(2**i), Sz), np.identity(2**(n-1-i)))
    Sx_total = Sx_total.dot(Sx_total)
    Sy_total = Sy_total.dot(Sy_total)
    Sz_total = Sz_total.dot(Sz_total)
    S2 = Sx_total + Sy_total + Sz_total
    H = Ham_xxz(1, np.array([1, 1, 1]), n, 'close')
    tup = (S2, H[1])
    [S2, new_basis] = H_BlockDiag(tup, n)
    ro = state_column.dot(state_line)
    s2_ro=S2.dot(ro)
    s2 = np.trace(s2_ro)
    return s2

        
def FormingHamiltonian(n, states_one_magnon, a_1, a_2, a_3, eig_vec3):
    length_2 = int(n*(n-1)/2)
    H_bethe = np.zeros([2**n, 2**n])
    H_bethe = H_bethe.astype(complex)
    H_bethe[0,0] = 1
    limit1 = int(n+1)
    H_bethe[1:limit1, 1:limit1] = states_one_magnon
    limit2 = limit1 + length_2
    H_bethe[limit1:limit2, limit1:limit1 + n] = a_1
    limit3 = limit1 + n + int(n*(n-5)/2 + 3)
    H_bethe[limit1:limit2, limit1 + n:limit3] = a_2
    H_bethe[limit1:limit2, limit3:limit2] = a_3
    limit4 = limit2 + n*(n-1)*(n-2)/6
    H_bethe[limit2:limit4,limit2:limit4] = eig_vec3    
    if n == 6:
        I=np.zeros([length_2, length_2])
        for i in range(length_2):
            I[i,-i-1] = 1
        H_2magnon_inv = I@H_bethe[limit1:limit2, limit1:limit2]@I       
        I=np.zeros([n, n])
        for i in range(n):
            I[i,-i-1] = 1
        H_1magnon_inv = I@H_bethe[1:1+n, 1:1+n]@I       
        H_bethe[42:57,42:57] = H_2magnon_inv
        H_bethe[57:63,57:63] = H_1magnon_inv
        H_bethe[63,63] = 1        
    return H_bethe
    
    
    
    
    
    