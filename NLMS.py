"""
Iterative normalized least-mean-squares (NLMS) algorithm for signal recovery.

"""

from __future__ import division
import numpy as np
import numpy.linalg as npl

from scipy.io import loadmat
from scipy.io.wavfile import write as wavwrite

################################################# MAIN

# Do you want to save and plot?
save_audio = False
plot_results = True


def regmat(x, m):
    """
    Returns the order-m filter regression matrix of a timeseries x.
    This is the matrix squareroot of the autocorrelation.

    """
    # Number of order-m lags of the signal that can be fit into itself
    nlags = len(x) - m

    # Row-stack each set of m data points
    X = np.zeros((nlags, m))
    for i in xrange(nlags):
        X[i, :] = x[i:(i+m)]
    return X


def nlms(x, y, m, a, z=1):
    """
    Returns the array of m weights of the (hopefully) MSE-optimal filter for a given input
    data array x and a desired output data array y. Also returns a list of the errors, approximate
    SNRs, and weights at each iteration. The algorithm used is gradient descent with stepsize a.
    (The filter order is m of course). The timeseries is iterated through z times (number of "epochs").

    """
    # Initialization
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    m = int(m); z = int(z)
    w = np.zeros(m)
    X = regmat(x, m)
    e_list = []; snr_list = []; w_list = []

    # Routine
    for run in xrange(z):
        for i in xrange(len(x) - m):
            w_list.append(w)
            xi = x[i:(i+m)]
            yi = y[i+m-1]
            e = yi - w.dot(xi)
            w = w + a*(e*xi)/(xi.dot(xi))
            e_list.append(e)
            if not i%100:
                snr_list.append((i, 10*np.log10(np.mean(np.square(y[m:(m+i+1)]))/np.mean(np.square(e_list[:i+1])))))
    return w, e_list, snr_list, w_list


# Unpack data
data = loadmat('audio_data.mat')
noisy_speech = data['reference'].flatten()
noise = data['primary'].flatten()
fs = data['fs'].flatten()  # Hz


# See http://www.cs.cmu.edu/~aarti/pubs/ANC.pdf
m = 100
a = 0.03
w, e_list, snr_list, w_list = nlms(noise, noisy_speech, m, a)
speech = np.array(e_list, dtype=np.float64)
se_arr = np.square(speech)
snr_arr = np.array(snr_list)
w_arr = np.array(w_list, dtype=np.float64)

################################################# RECORD

if save_audio:
    wavwrite('recovered_NLMS.wav'.format(m), fs, speech)

################################################# VISUALIZATION

if plot_results:

    # More imports for plotting
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as plt3
    import matplotlib.cm as cm
    fontsize = 30

    # Performance contour
    fig1 = plt.figure()
    fig1.suptitle('Performance Contour (order: {}, stepsize: {})'.format(m, a), fontsize=fontsize)
    ax1 = plt3.Axes3D(fig1)
    ax1.set_xlabel('Weight 1', fontsize=fontsize)
    ax1.set_ylabel('Weight 2', fontsize=fontsize)
    ax1.set_zlabel('Square Error', fontsize=fontsize)
    ax1.grid(True)
    ax1.plot(w_arr[:, 0], w_arr[:, 1], se_arr)

    # Weight tracks
    fig2 = plt.figure()
    fig2.suptitle('Weight Tracks (order: {}, stepsize: {})'.format(m, a), fontsize=fontsize)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlabel('Iteration', fontsize=fontsize)
    ax2.set_ylabel('Weight Value', fontsize=fontsize)
    ax2.set_ylim((-3, 3))
    ax2.grid(True)
    ax2.plot(w_arr)

    # Learning curve
    fig2 = plt.figure()
    fig2.suptitle('Learning Curve (order: {}, stepsize: {})'.format(m, a), fontsize=fontsize)
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.set_xlabel('Iteration', fontsize=fontsize)
    ax2.set_ylabel('Square Error', fontsize=fontsize)
    ax2.set_ylim((0, 50))
    ax2.grid(True)
    ax2.plot(se_arr)

    # SNR Iteration
    fig3 = plt.figure()
    fig3.suptitle('Approximate SNR (order: {}, stepsize: {})'.format(m, a), fontsize=fontsize)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_xlabel('Iteration', fontsize=fontsize)
    ax3.set_ylabel('ERLE (dB)', fontsize=fontsize)
    ax3.grid(True)
    ax3.plot(snr_arr[:, 0], snr_arr[:, 1])

    plt.show()
