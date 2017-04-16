"""
Sliding-analytical least-mean-squares (SLMS) algorithm for signal recovery.
This is the ALMS with a sliding window carried out iteratively.

"""
from __future__ import division
import numpy as np
import numpy.linalg as npl

from scipy.linalg import toeplitz
from scipy.io import loadmat
from scipy.io.wavfile import write as wavwrite

# Function for forming filter regression matrix given an array (data) and order (m)
regmat = lambda data, m: toeplitz(data, np.concatenate(([data[0]], np.zeros(m-1))))[m-1:]

# Unpack data
data = loadmat('audio_data.mat')
noisy_speech = data['reference'].flatten()
noise = data['primary'].flatten()
fs = data['fs'].flatten()[0]  # Hz

# Filter order
m = 100

# Window size must be larger than filter order
s = 10*m
assert s > m

# Prep
X = regmat(noise, m)
w_list = []
speech = []

# Slide window by window through the input data
# and find the analytically optimal weights for
# the output data, keeping the speech signal
# as the error for each sample in that window
for i in np.arange(0, len(noise)-s, s):
    x = X[i:(i+s)]
    y = noisy_speech[(i+m-1):(i+m-1+s)]
    w = npl.pinv(x).dot(y)
    e = y - x.dot(w)
    w_list.append(w)
    speech.extend(e)

# Save results
wavwrite('recovered_SLMS.wav'.format(m, s), fs, np.array(speech))
