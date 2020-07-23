# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:51:28 2020

@author: Alexander Mitrofanov
"""


import numpy as np
import math
from scipy.optimize import fsolve
from scipy.optimize import newton_krylov
from scipy.optimize import anderson

def clean(A, order):
    A = A.astype(complex)
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
        A = A.astype(complex)
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
    k_1 = fsolve(func, k_initial_guess, xtol=1e-12, maxfev = 10000)
    # k_1 = newton_krylov(func, k_initial_guess, f_tol=1e-12)
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
            nu_initial_guess = 1
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
    Sx =0.5*np.array([[0, 1], [1, 0]])
    Sy =0.5*np.array([[0, -1j], [1j, 0]])
    Sz =0.5*np.array([[1, 0], [0, -1]])
    if n == 4:
        Sx2 = np.kron(np.identity(8), Sx) + np.kron(np.kron(np.identity(2), Sx), np.identity(4)) + np.kron(np.kron(np.identity(4), Sx), np.identity(2)) + np.kron(Sx, np.identity(8))
        Sy2 = np.kron(np.identity(8), Sy) + np.kron(np.kron(np.identity(2), Sy), np.identity(4)) + np.kron(np.kron(np.identity(4), Sy), np.identity(2)) + np.kron(Sy, np.identity(8))
        Sz2 = np.kron(np.identity(8), Sz) + np.kron(np.kron(np.identity(2), Sz), np.identity(4)) + np.kron(np.kron(np.identity(4), Sz), np.identity(2)) + np.kron(Sz, np.identity(8))
        Sx2 = Sx2 ** 2
        Sy2 = Sy2 ** 2
        Sz2 = Sz2 ** 2
        S2 = Sx2 + Sy2 + Sz2
        ro = state.dot(np.transpose(state))
        s2 = np.trace(ro)
        return s2

        
    
    
    
    
    
    
    