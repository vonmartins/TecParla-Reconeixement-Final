#!/usr/bin/env python3

import numpy as np
from python_speech_features import mfcc

def extraer_mfcc(sen, sr=8000, numcep=13, nfilt=26):
    # Extraer MFCC usando la libreria
    coefs = mfcc(sen, samplerate=sr, numcep=numcep, nfilt=nfilt)
    # Devolver la media de todos los frames
    return coefs.mean(axis=0)
