#!/usr/bin/env python3

import numpy as np

def autocorr(x, p):
    # Autocorrelacion
    r = np.correlate(x, x, mode='full')
    r = r[len(r)//2:]
    return r[:p+1]

def levinson_durbin(r, p):
    # Algoritmo de Levinson-Durbin
    a = np.zeros(p+1)
    a[0] = 1.0
    e = r[0]
    
    for i in range(1, p+1):
        lambda_val = 0
        for j in range(i):
            lambda_val += a[j] * r[i-j]
        k = -lambda_val / e
        
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i-j]
        a_new[i] = k
        
        a = a_new
        e = e * (1 - k**2)
    
    return a[1:], e

def extraer_maxent(sen, p=12, nfft=512):
    # Calcula autocorrelacion
    r = autocorr(sen, p)
    # Coeficientes LPC
    a, e = levinson_durbin(r, p)
    
    # Calcular espectro
    A = np.fft.fft(np.concatenate([[1], -a]), nfft)
    spectrum = e / (np.abs(A)**2)
    
    return 10 * np.log10(spectrum[:nfft//2])
