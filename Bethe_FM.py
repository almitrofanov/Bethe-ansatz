# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:52:31 2020

@author: Alexander Mitrofanov
"""

import numpy as np
import math
from Helpers import *

def one_magnon(J, n):
    E0 = -J*n
    a = np.zeros([n,n])
    a = a.astype(complex)
    E_1magnon = np.zeros([n, 1])
    E_1magnon = E_1magnon.astype(complex)
    for i in range(n):
        k = 2*math.pi*i/n
        E_1magnon[i] = 4*(1 - math.cos(k)) + E0
        for j in range(n):
            a[j, i] = np.exp(1j*k*j)
    a = clean(a, 1e-10)
    E_1magnon = clean(E_1magnon, 1e-10)
    E_1magnon = E_1magnon.astype(float)
    b = normalyze(a)
    return b, E_1magnon

def two_magnons(J, n):
    # the first class C1 (lambda1=0, lambda2 = 0:n-1, k1=0, k2 = 2*pi*lambda2/n, theta=0)
    length = int(n*(n-1)/2)
    a_1 = np.zeros([length,n])
    a_1 = a_1.astype(complex)
    E_1 = np.zeros([n,1])
    for i in range(n):
        lambd = [0, i]
        k = [0, 2*math.pi*lambd[1]/n]
        theta = 0
        count = 0
        for n1 in range(1, n):
            for n2 in range(n1+1, n+1):
                a_1[count, i] = (np.exp(1j * (k[0] * n1 + k[1]*n2 + 0.5*theta))) + (np.exp(1j * (k[0] * n2 + k[1]*n1 - 0.5*theta))) 
                count = count + 1
        E_1[i] = 4*J*(1 - math.cos(k[1])) -J*n
    a_1 = normalyze(a_1)
    a_1 = clean(a_1, 1e-10)
    # E_1 = clean(E_1, 1e-10)
    # E_1 = int(np.real(E_1))
    
    # the second class C2 (lambda2-lambda1 >= 2,  k1, k2 and theta != 0)
    count2 = 0
    a_2 = np.zeros([length, int(n*(n-5)/2 + 3)])
    a_2 = a_2.astype(complex)
    E_2 = []
    for i in range(1, n-1):
        for j in range(i+2, n):
            if (i != 1) or (j != n):
                lambd = [i,j]
                # print(lambd)
                k_0 = 2*np.pi*(lambd[0] + lambd[1])/n
                [k_1, k_2, theta] = solving_C2(k_0, lambd, n)
                count1 = 0
                for n1 in range(1, n):
                    for n2 in range(n1+1, n+1):
                        a_2 [count1, count2] =  (np.exp(1j*(k_1*n1+k_2*n2+0.5*theta))+np.exp(1j*(k_1*n2+k_2*n1-0.5*theta)))
                        count1 = count1 + 1                    
                E_2.append(4*J*(1 - math.cos(k_1)) + 4*J*(1 - math.cos(k_2)) -J*n)
                count2 = count2 + 1
    a_2 = normalyze(a_2)
    a_2 = clean(a_2, 1e-10)      
     
    # C3 (lambda1=lambda2=1, k1=pi/2+1j*inf, k2=pi/2-1j*inf,) 
    if n == 4:
        count = 0
        a_3 = np.zeros([length, 1])
        a_3 = a_3.astype(complex)
        for n1 in range(1, n):
            for n2 in range(n1+1, n+1):
                deltaFunc = DeltaFunction(n2, n1+1)
                a_3[count, 0] = 1j*((-1)**n1)*deltaFunc
                if n1 ==1 and n2 ==n:
                    a_3[count, 0] = 1j
                count = count + 1
        a_3 = normalyze(a_3)
        a_3 = clean(a_3, 1e-10)
        E_3 = 0       
    else:
        # length = int(n*(n-1)/2)
        count = 0
        a_3 = np.zeros([length, n-3])
        a_3 = a_3.astype(complex)
        count2 = 0
        E_3 = np.zeros([n-3])
        E_3 = E_3.astype(complex)
        for i in range(1, n):
            for j in range(i, i+2):
                lambd = [i,j]
                
                k_0 = 2*np.pi*(lambd[0] + lambd[1])/n
                [k_1, k_2, theta] = solving_C3(k_0, lambd, n)
                # print(k_1, k_2, theta)
                if theta != np.inf:
                    print(lambd)
                    count1 = 0
                    for n1 in range(1, n):
                        for n2 in range(n1+1, n+1):
                            a_3[count1, count2] = (np.exp(1j*(k_1*n1+k_2*n2+0.5*theta))+np.exp(1j*(k_1*n2+k_2*n1-0.5*theta)))
                            count1 = count1 + 1
                    E_3[count2] =  8*J*(1-np.cos(k_1.real)*np.cosh(k_1.imag))-J*n   
                    # print(E_3)
                    count2 = count2 + 1                                    
        
        if np.linalg.norm(a_3[:, n-4]) == 0:
            count1 = 0
            for n1 in range(1,n):
                for n2 in range(n1+1,n+1):
                    deltaFunc = DeltaFunction(n2, n1+1)
                    a_3[count1, count2] = 1j*((-1)**n1)*deltaFunc
                    if n1 == 1 and n2 == n:
                        a_3[count1, count2] = 1j
                    count1 = count1 + 1
                    E_3[count2] = - ( 8*J*(1-np.cos((np.pi)/2))-J*n) # possible mistake?
        a_3 = normalyze(a_3)
        a_3 = clean(a_3, 1e-10)
    a_1= ChangeOrder(a_1)
    a_2= ChangeOrder(a_2)
    a_3= ChangeOrder(a_3)
    return a_1, a_2, a_3, E_1, E_2, E_3


    
    
    
    