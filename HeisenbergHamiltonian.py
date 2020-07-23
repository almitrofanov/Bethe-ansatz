# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:10:35 2020

@author: urazhdinlabuser
"""
import numpy as np

def Ham_xxz(J, J_a, n, border):
    
    Jx=J*J_a[0]
    Jy=J*J_a[1]
    Jz=J*J_a[2]
        
    Sx=np.array([(0, 1),  (1, 0)])
    Sy=np.array([(0, -1j),  (1j, 0)])
    Sz=np.array([(1, 0),  (0, -1)])

        
    basis = getBasisVectors(n)
    
    H = -Jz*np.kron(Sz, Sz)-Jx*np.kron(Sx, Sx)-Jy*np.kron(Sy, Sy)
    
    for i in range(n-2):
        i=i+1
        Sx_tilde = np.kron(np.identity(2**i), Sx)
        Sy_tilde = np.kron(np.identity(2**i), Sy)
        Sz_tilde = np.kron(np.identity(2**i), Sz)
        H= np.kron(H, np.identity(2)) -np.kron(Sz_tilde, Sz)-Jx*np.kron(Sx_tilde, Sx)-Jy*np.kron(Sy_tilde, Sy)
       
    if border == 'close':
        Sx_tilde = np.kron(Sx, np.identity(2**(n-2)))
        Sy_tilde = np.kron(Sy, np.identity(2**(n-2)))
        Sz_tilde = np.kron(Sz, np.identity(2**(n-2)))
        H=H-np.kron(Sz_tilde, Sz)-Jx*np.kron(Sx_tilde, Sx)-Jy*np.kron(Sy_tilde, Sy)

    return H, basis
 
    

def getBasisVectors(n):
    basis = []
    for i in range(2**n):
        k = bin(2**n-1-i)
        base = np.zeros(n).astype(int)
        for i in range(len(k)-2):
            base[-i-1] = int(k[-i-1])
        basis.append(base)
        
    return basis