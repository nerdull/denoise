#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
from scipy.linalg import svd


class Denoise(object):
    '''
    A class for smoothing a noisy, real-valued data sequence by means of SVD of a partial circulant matrix.
    '''

    def __init__(self, sequence, detrend="constant", aggregation=9):
        '''
        Class initialization.
        arguments:
            sequence:    noisy data sequence
            detrend:     if "constant", remove the DC level from the data
                         if "linear", remove the inclination from the data
                         default is "constant"
            aggregation: number of neighboring data points used for average to estimate the boundary levels of the sequence
                         an odd integer is recommended
                         default is 9
        '''
        self.seq = sequence
        self.n = sequence.size
        if detrend not in ["constant", "linear"]:
            raise ValueError("Unknown detrend type '{:s}'!".format(detrend))
        else:
            self.detrend = detrend
            self.trend = {"constant": np.mean(sequence), "linear": None}
        self.aggr = aggregation

    def _embed(self, a, m):
        '''
        Embed a 1D array into a 2D partial circulant matrix by cyclic shift.
        arguments:
            a: input array
            m: number of rows of the matrix
        returns:
            A: partial circulant matrix
        '''
        a_ext = np.hstack((a, a[:m-1]))
        shape = (m, self.n)
        strides = (a_ext.strides[0], a_ext.strides[0])
        A = np.lib.stride_tricks.as_strided(a_ext, shape, strides)
        return A

    def _crush(self, A):
        '''
        Crush a 2D matrix to a 1D array by cyclic anti-diagonal average.
        arguments:
            A: input matrix
        returns
            a: output array
        '''
        m = A.shape[0]
        A_ext = np.hstack((A[:,-m+1:], A))
        strides = (A_ext.strides[0]-A_ext.strides[1], A_ext.strides[1])
        a = np.mean(np.lib.stride_tricks.as_strided(A_ext[:,m-1:], A.shape, strides), axis=0)
        return a

    def denoise(self, layer):
        '''
        Smooth the sequence by discarding the noise components after singular value decomposition.
        arguments:
            layer:    number of leading rows selected from a circulant matrix, which is formed from the sequence
        returns:
            denoised: smoothed sequence after denoise
        '''
        assert 1 <= layer <= self.n
        begin, end = np.mean(self.seq[:self.aggr]), np.mean(self.seq[-self.aggr:])
        self.trend["linear"] = np.arange(self.n) * (end-begin) / (self.n-1) + begin
        detrended = self.seq - self.trend[self.detrend]
        A = self._embed(detrended, layer)
        U, s, Vh = svd(A, full_matrices=False, overwrite_a=True, check_finite=False)
        # search for noise components using the normalized total variation of the left singular vectors as an indicator
        # the procedure runs in batch of every 10 singular vectors
        index = 0
        while index < layer:
            U_sub = U[:,index:index+10]
            total_var = np.mean(np.abs(np.diff(U_sub,axis=0)), axis=0) / (np.amax(U_sub,axis=0) - np.amin(U_sub,axis=0))
            try:
                # the threshold of 10% can in general discriminate noise components
                index += np.argwhere(total_var > .1)[0,0]
                break
            except IndexError:
                index += 10
        # estimate the noise strength, while index marks the first noise component
        noise_stdev = np.sqrt(np.sum(s[index:]**2) / layer / self.n)
        # estimate the boundary gap after detrend
        gap = np.mean(detrended[-self.aggr:]) - np.mean(detrended[:self.aggr])
        if np.abs(gap) < noise_stdev: # boundary gap within noise strength validates the detrend procedure
            # approximate A using only signal components
            A_s = U[:,:index] @ np.diag(s[:index]) @ Vh[:index,:]
            denoised = self._crush(A_s) + self.trend[self.detrend]
            return denoised
        else: # otherwise, a residual linear trend is still in effect
            if self.detrend == "constant":
                self.detrend = "linear"
                return self.denoise(layer)
            else: # in the worst case, recursion terminates when aggregation is 1
                self.aggr -= 2
                return self.denoise(layer)


if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    signal = np.sinc(x)
    noise = np.random.normal(scale=.1, size=1000)
    sequence = signal + noise
    denoise = Denoise(sequence)
    denoised = denoise.denoise(200)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x, sequence)
    ax.plot(x, signal)
    ax.plot(x, denoised)
    plt.show()
