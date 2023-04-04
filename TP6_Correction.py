"""
installer les bib suivantes :
    
pip install python_speech_features
pip install sounddevice
pip install soundfile

"""
import numpy as np
import matplotlib.pyplot as plt

import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write

from python_speech_features import mfcc, delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

''' recording wave speech signal '''

# fs = 16000  # Sample rate
# seconds = 3  # Duration of recording
# print('start recording')
# x = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, x)  # Save as WAV file 

# sd.play(x, fs)
# status = sd.wait()  # Wait until file is done playing
# print('fin recording')

''' reading wave speech signal '''

(fs,x) = wav.read("SA1.wav")
sd.play(x, fs)

time=np.arange(0,len(x))/fs; # Time vector on x-axis
plt.figure()
plt.plot(time, x, label="signal wav")
plt.title('signal wav')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
# plt.show()

''' feature extraction using MFCC'''

mfcc_feat = mfcc(x,fs)
fbank_feat = logfbank(x,fs)

mfccs=mfcc_feat

# print(mfcc_feat[1:3,:])
# print(fbank_feat[1:3,:])

plt.figure()
plt.imshow(mfccs.T, origin='lower', aspect='auto', interpolation='nearest')
plt.ylabel('MFCC Coefficient Index')
plt.xlabel('Frame Index')
