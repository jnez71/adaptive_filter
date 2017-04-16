"""
Analytical least-mean-squares (ALMS) algorithm for signal recovery.

"""

from __future__ import division
import numpy as np
import numpy.linalg as npl

from scipy.linalg import toeplitz
from scipy.io import loadmat
from scipy.io.wavfile import write as wavwrite

################################################# MAIN

# Do you want to save and plot?
save_audio = False
plot_results = True

# Function for forming filter regression matrix given an array (data) and order (m)
regmat = lambda data, m: toeplitz(data, np.concatenate(([data[0]], np.zeros(m-1))))[m-1:]

# Unpack data
data = loadmat('audio_data.mat')
noisy_speech = data['reference'].flatten()
noise = data['primary'].flatten()
fs = data['fs'].flatten()  # Hz

# Filter order
m = 100

# See http://www.cs.cmu.edu/~aarti/pubs/ANC.pdf page 7 last paragraph
X = regmat(noise, m)
y = noisy_speech[m-1:]
w = npl.pinv(X).dot(y)

# The error is the signal we seek
speech = y - X.dot(w)

################################################# RECORD

if save_audio:
    wavwrite('recovered_ALMS.wav'.format(m), fs, speech)

################################################# VISUALIZATION

if plot_results:
    
    # Compute performance surface for order 2
    m2 = 2
    X2 = regmat(noise, m2)
    y2 = noisy_speech[m2-1:]
    w2 = npl.pinv(X2).dot(y2)
    print("Optimal weights for order 2: {}".format(w2))
    w1_arr = np.arange(-10, 10)
    w2_arr = np.arange(-10, 10)
    trio = []
    for i in w1_arr:
        for j in w2_arr:
            trio.append([i, j, np.mean(np.square(y2 - X2.dot([i, j])))])

    # More imports for plotting
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as plt3
    import matplotlib.cm as cm
    fontsize = 30

    # Performance surface
    fig1 = plt.figure()
    fig1.suptitle('Uniform Performance Surface (order: {})'.format(m2), fontsize=fontsize)
    ax1 = plt3.Axes3D(fig1)
    ax1.set_xlabel('Weight 1', fontsize=fontsize)
    ax1.set_ylabel('Weight 2', fontsize=fontsize)
    ax1.set_zlabel('Mean Square Error', fontsize=fontsize)
    ax1.grid(True)
    trio = np.array(trio)
    ax1.plot_trisurf(trio[:, 0], trio[:, 1], trio[:, 2], cmap=cm.jet, linewidth=0.2)
    ax1.plot([w[0]]*1000, [w[1]]*1000, np.linspace(0, np.max(trio[:, 2]), 1000), color='r', linewidth=1)

    plt.show()
