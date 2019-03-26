import numpy as np
import scipy
import data_acquisition as voice
import preprocessing
import featureExtraction as fe
import smoothen
import sigproc
from transferFunction import transfer_function
import scipy.io as sio
seed = sio.loadmat('../seed1250.mat')

audio = seed['audio'][0:2]

Fs = 11025
smooth_factor = 1

filtered_audio = preprocessing.preprocess(audio, Fs)
features_test = fe.extract(filtered_audio,Fs)

print(features_test.shape)
