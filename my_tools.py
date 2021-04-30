# -*- coding: utf-8 -*-
#
# my_tools.py
#
# Copyright (c) 2014 Dominique Fourer <dominique@fourer.fr>

# This file is part of TimeSide.

# TimeSide is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# TimeSide is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TimeSide.  If not, see <http://www.gnu.org/licenses/>.

# Author: D Fourer <dominique@fourer.fr> http://www.fourer.fr

import numpy
import scipy
#import matplotlib
#import matplotlib.pyplot as plt
#import ipdb


EPS = 2.2204 * pow(10, -16.0)


def sub_matrix(a, I1, I2):
    """
        Compute submatrix
    """
    return a.take(I1, axis=0).take(I2, axis=1)


def Index_to_freq(k, sr_hz, sizeFFT):
    """
        DFT bin to frequency conversion
        replace < [f] = Fmatlabversf(k, sr_hz, sizeFFT) >
        name: unknown
        @param
        @return
    """
    return k / sizeFFT*sr_hz


def nextpow2(x):
    return numpy.ceil(numpy.log2(x) + EPS)


def my_max(vec):
    vm = numpy.max(vec)
    if numpy.isnan(vm):
        im = 0
    else:
        im = (vec == vm).nonzero()[0][0]
    return vm, im


def my_min(vec):
    vm = numpy.min(vec)
    if numpy.isnan(vm):
        im = 0
    else:
        im = (vec == vm).nonzero()[0][0]
    return vm, im


def my_sort(x, order='ascend'):
    I = numpy.argsort(x)
    if order == 'descend':
        I = I[::-1]
    x_hat = x[I]
    return x_hat, I


def my_linspace(deb, fin, step):
    out_val = numpy.array([deb])
    last_e = out_val[len(out_val)-1]
    while last_e+step+EPS < fin:
        last_e = last_e+step
        out_val = numpy.concatenate((out_val, [last_e]))

    out_val = numpy.concatenate((out_val, [fin]))
    #nb_case = round((fin-deb)/step)+1
    return out_val #numpy.linspace(deb, fin, nb_case);


def xcorr(vec):
    """
        Equivalent to xcorr(vec, 'coeff')
        name: unknown
        @param
        @return
    """
    n = len(vec)-1
    if n < 0:
        return []
    cor = numpy.zeros(2 * n + 1, float)
    for idx in range(0, n):
        m = n-idx
        vtmp2 = vec[m:]
        vtmp1 = vec[0:len(vtmp2)]    ## bug ?
        cor[idx] = numpy.dot(vtmp1, vtmp2) #numpy.sum(
        cor[len(cor)-idx-1] = cor[idx]
    ##print "TAILLE !! : ", len(vec);
    cor[n] = numpy.dot(vec, vec)
    cor = cor / cor[n]
    return cor
    #return numpy.real( scipy.ifft(scipy.fft(vec, 2*n+1) * scipy.fft(vec, 2*n+1) ))  ;


## used for debug only
def disp_var(x, name="var", shape=True, plot=True):
    print(name, " : ")
    #print x, "\n";
    print(sum(abs(x)), "\n")

    if shape:
        print(name, " shape : ", numpy.shape(x), "\n")
    if plot:
        my_plot(x, [], "X", "Y", name)


def my_plot(y, x=[], xlab="X axis", ylab="Y axis", title="Figure"):
    """
    My plotting function
    name: unknown
    @param
    @return
    """
    is_complex = numpy.iscomplex(y).any()

    y_old = y
    if is_complex and len(x) != len(y):
        x = numpy.real(y_old)
        y = numpy.imag(y_old)

    if len(x) != len(y):
        plt.plot(y)
    else:
        plt.plot(x, y)

    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
    return


def rms(s,N=0,rec=2,Fs=44100):
    """
        Compute RMS energy of input signal
        name: rms
        @param
        @return
    """
    #s = numpy.array(s);

    if N == 0 or N > len(s):
        N = len(s)
    if N == len(s):
        v = numpy.sqrt(numpy.mean(s ** 2))
        t = numpy.arange(1, len(s)+1)
    else:
        hop = numpy.floor(N / rec)
        nb_trame = int(numpy.ceil(len(s) / hop))
        v = numpy.zeros(nb_trame, float)
        t = numpy.zeros(nb_trame, float)
        for i in range(nb_trame):
            i0 = i * hop
            i1 = min_(i0 + N - 1, len(s))
            t0 = (i0 + i1) / 2.0
            t[i] = t0 / Fs
            v[i] = numpy.sqrt(numpy.mean(s[i0:(i1+1)] ** 2.0))
    return v, t


def IQR(x):
    return numpy.percentile(x, 75) - numpy.percentile(x, 25)


def hz2erbs(hz):
    return 6.44 * (numpy.log2(229. + hz ) - 7.84)


def erbs2hz(erbs):
    return pow(2, erbs / 6.44 + 7.84 ) - 229


def my_interpolate(x, y, xi, kind='linear'):
    f = scipy.interpolate.interp1d(x, y, kind, 0, False, 0.0)
    y_tmp = f(xi)
    return y_tmp


def primes(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = numpy.ones(int(n/3 + (n % 6 == 2)), dtype=numpy.bool)
    for i in range(1, int(int(n**0.5)/3+1) ):
        if sieve[i]:
            k = 3*i+1 | 1
            sieve[ int(k*k/3):: int(2*k)] = False
            sieve[ int(k*(k-2*(i & 1)+4)/3) :: int(2*k)] = False
    return numpy.r_[2, 3, ((3*numpy.nonzero(sieve)[0][1:]+1) | 1)]


def specgram(x, nfft, Fs, window, noverlap):
    if nfft != len(window):
        padding = nfft
        nfft = len(window)
    else:
        padding = None
    B_m, F_v, T_v = matplotlib.mlab.specgram(x, nfft, Fs, matplotlib.mlab.detrend_none, window, noverlap, padding, 'default', False) #False #''default'
    T_v = T_v - T_v[0]
    return B_m, F_v, T_v


def my_specgram(x, nfft, Fs, window, noverlap):
    if nfft != len(window):
        padding = nfft
        nfft = len(window)
    else:
        padding = len(window)

    hop = float(nfft - noverlap + 1)
    nb_trame = int(numpy.floor(float((len(x)-noverlap) / hop)))

    step = hop/float(Fs)
    T_v = numpy.arange(0, nb_trame, 1.) * step
    F_v = numpy.arange(0, padding, 1.) / float(padding) * float(Fs)
    B_m = numpy.zeros((padding, len(T_v)), complex)

    for i in range(0, nb_trame):
        i0 = int(i * hop)
        i1 = min(len(x), i0+nfft-1)
        trame = numpy.zeros(nfft, float)
        trame[0:len(x[i0:i1])] = x[i0:i1]
        B_m[:, i] = scipy.fft(trame * window, padding)

    if ~numpy.iscomplex(x).any():
        Hs = int(numpy.round(padding/2+1))
        F_v = F_v[0:Hs]
        B_m = B_m[0:Hs, :]

    return B_m, F_v, T_v


def centroid(x):
    if numpy.ndim(x) < 2:
        x = numpy.array([x, ])
    m, n = numpy.shape(x)
    if m < n:
        x = x.T
        m, n = numpy.shape(x)
    M = numpy.array([numpy.arange(1, m+1, 1.), ])
    idx = numpy.repeat(M.T, n, axis=0)
    c = numpy.sum(x * idx, axis=0)
    w = numpy.sum(x, axis=0)
    return c / (w + EPS)


def ERB(cf):
    return 24.7 * (1. + 4.37*cf/1000.)


def ERBfromhz(f, formula='glasberg90'):
    """
        e=ERBfromhz(f,formula) - convert frequency from Hz to erb-rate scale
        f: Hz - frequency
        formula: 'glasberg90' [default] or 'moore83'
    """
    if formula == 'glasberg90':
        e = 9.26 * numpy.log(0.00437*f + 1.)
    else: #'moore83'
        erb_k1 = 11.17
        erb_k2 = 0.312
        erb_k3 = 14.675
        erb_k4 = 43
        f = f/1000.
        e = erb_k1 * numpy.log((f + erb_k2) / (f + erb_k3)) + erb_k4
    return e


def gtwindow(n, b=2., order=4.):
    """
        y=gtwindow(n, b, order) - window shaped like a time-reversed gammatone envelope
    """
    t = numpy.arange(0, n, 1.) / n
    y = pow(b, order) * pow(t, (order-1)) * numpy.exp(-2. * numpy.pi * b * t)
    y = numpy.flipud(y)
    y = y/numpy.max(y)
    return y


def ERBspace(lo, hi, N=100):
    """
        y = ERBspace(lo,hi,N) - values uniformly spaced on  erb-rate scale
    """

    EarQ = 9.26449
    minBW = 24.7
    order = 1
    a = EarQ * minBW
    cf = -a + numpy.exp((numpy.arange(0, N, 1.))*(-numpy.log(hi + a) + numpy.log(lo + a))/(N-1)) * (hi + a)
    y = numpy.squeeze(numpy.fliplr([cf, ]))
    return y


def frames(a, fsize, hopsize, nframes=0):
    """
        b=frames(a,fsize,hopsize,nframes) - make matrix of signal frames
        b: matrix of frames (columns)
        a: signal vector
        fsize: samples - frame size
        hopsize: samples - interval between frames
        nframes: number of frames to return (default: as many as fits)

        hopsize may be fractional (frame positions rounded to nearest integer multiple)
    """
    n = len(a)
    max_nf = numpy.ceil((n-fsize)/hopsize)
    if (nframes == 0) or (nframes > max_nf):
        nframes = int(max_nf[0])
    b = numpy.ones((fsize, nframes), int)
    t = 1. + numpy.round(hopsize * numpy.arange(0, nframes, 1.))
    b[0, :] = t
    b = numpy.cumsum(b, axis=0)
    a = numpy.concatenate((a, numpy.zeros(fsize, float)))
    b = a[b]
    return b, t


def gtfbank(a, Fs, cfarray, bwfactor):
    """
    """
    lo = 30.
    hi = 16000.
    hi = min(hi, (Fs / 2. - numpy.squeeze(ERB(Fs/2.))/2.))
    nchans = numpy.round(2. * (ERBfromhz(hi) - ERBfromhz(lo)))
    cfarray = ERBspace(lo, hi, nchans)

    nchans = len(cfarray)
    n = len(a)

    # make array of filter coefficients
    fcoefs = MakeERBCoeffs(Fs, cfarray, bwfactor)
    A0 = fcoefs[:, 0]
    A11 = fcoefs[:, 1]
    A12 = fcoefs[:, 2]
    A13 = fcoefs[:, 3]
    A14 = fcoefs[:, 4]
    A2 = fcoefs[:, 5]
    B0 = fcoefs[:, 6]
    B1 = fcoefs[:, 7]
    B2 = fcoefs[:, 8]
    gain = fcoefs[:, 9]

    b = numpy.zeros((nchans, n), float)
    output = numpy.zeros((nchans, n), float)
    for chan in range(0, nchans):
        B = numpy.concatenate(([A0[chan]/gain[chan], ], [A11[chan]/gain[chan], ], [A2[chan]/gain[chan], ]))
        A = numpy.concatenate(([B0[chan], ], [B1[chan], ], [B2[chan], ]))
        y1 = scipy.signal.lfilter(B, A, a)

        B = numpy.concatenate(([A0[chan], ], [A12[chan], ], [A2[chan], ]))
        A = numpy.concatenate(([B0[chan], ], [B1[chan], ], [B2[chan], ]))
        y2 = scipy.signal.lfilter(B, A, y1)

        B = numpy.concatenate(([A0[chan], ], [A13[chan], ], [A2[chan], ]))
        A = numpy.concatenate(([B0[chan], ], [B1[chan], ], [B2[chan], ]))
        y3 = scipy.signal.lfilter(B, A, y2)

        B = numpy.concatenate(([A0[chan], ], [A14[chan], ], [A2[chan], ]))
        A = numpy.concatenate(([B0[chan], ], [B1[chan], ], [B2[chan], ]))
        y4 = scipy.signal.lfilter(B, A, y3)
        b[chan, :] = y4
    return b #f


def fbankpwrsmooth(a, Fs, cfarray):
    nchans = len(cfarray)
    # index matrix to apply pi/2 shift at each cf:
    shift = numpy.round(Fs / (4.*cfarray)).astype(int)
    a = pow(a, 2.)
    b = a * 0
    for j in range(0, nchans):
        b[j, :] = a[j, :] + numpy.concatenate((a[j, shift[j]:], numpy.zeros(shift[j], float))) / 2.
    return b


def rsmooth(x, smooth, npasses, clip=0):
    smooth = int(smooth)
    m, n = numpy.shape(x)

    mm = m + smooth + npasses*(smooth-1)
    a = numpy.zeros((mm, n), float)
    b = numpy.zeros((mm, n), float)
    tmp = 0

    # transfer data to buffer, preceded with leading zeros */
    a[smooth:(smooth+m), :] = x

    # smooth repeatedly */
    for h in range(0, npasses):
        for k in range(0, n):
            b[smooth:mm, k] = numpy.cumsum(a[smooth:mm, k] - a[0:(mm-smooth), k]) / smooth
        tmp = a
        a = b
        b = tmp

    if clip > 0:
        y = numpy.zeros((m, n), float)
        mm2 = int(npasses*(smooth-1)/2. + smooth)
        y[:, :] = a[mm2:(mm2+m), :]
    else:
        mm2 = m + npasses*(smooth-1)
        y = numpy.zeros((mm2, n), float)
        y[:, :] = a[smooth:mm2, :]
    return y


def MakeERBCoeffs(Fs, cfArray=0, Bfactor=1.):
    if len(cfArray) < 1:
        cfArray = ERBSpace(100., Fs/4, 25)
    T = 1./Fs
    EarQ = 9.26449
    minBW = 24.7
    order = 1
    cf = cfArray

    vERB = pow(pow(cf/EarQ, order) + pow(minBW, order), 1./order)
    B = 1.019 * 2 * numpy.pi * vERB * Bfactor

    A0 = T
    A2 = 0
    B0 = 1
    B1 = -2 * numpy.cos(2*cf*numpy.pi*T) / numpy.exp(B*T)
    B2 = numpy.exp(-2*B*T)

    A11 = -(2*T*numpy.cos(2*cf*numpy.pi*T) / numpy.exp(B*T) + 2*numpy.sqrt(3+pow(2,1.5))*T*numpy.sin(2*cf*numpy.pi*T) / numpy.exp(B*T))/2.
    A12 = -(2*T*numpy.cos(2*cf*numpy.pi*T) / numpy.exp(B*T) - 2*numpy.sqrt(3+pow(2,1.5))*T*numpy.sin(2*cf*numpy.pi*T) / numpy.exp(B*T))/2.
    A13 = -(2*T*numpy.cos(2*cf*numpy.pi*T) / numpy.exp(B*T) + 2*numpy.sqrt(3-pow(2,1.5))*T*numpy.sin(2*cf*numpy.pi*T) / numpy.exp(B*T))/2.
    A14 = -(2*T*numpy.cos(2*cf*numpy.pi*T) / numpy.exp(B*T) - 2*numpy.sqrt(3-pow(2,1.5))*T*numpy.sin(2*cf*numpy.pi*T) / numpy.exp(B*T))/2.

    gain = numpy.abs((-2*numpy.exp(4*1j*cf*numpy.pi*T)*T + 2. * numpy.exp(-(B*T) + 2*1j*cf*numpy.pi*T) * T * (numpy.cos(2. * cf * numpy.pi * T) - numpy.sqrt(3 - pow(2,3./2.)) * numpy.sin(2*cf*numpy.pi*T))) * (-2. * numpy.exp(4. * 1j * cf * numpy.pi*T)*T + 2. * numpy.exp(-(B*T) + 2*1j*cf*numpy.pi*T) * T * (numpy.cos(2.*cf*numpy.pi*T) + numpy.sqrt(3. - pow(2,3./2.)) * numpy.sin(2*cf*numpy.pi*T))) * (-2.*numpy.exp(4.*1j*cf*numpy.pi*T)*T + 2. * numpy.exp(-(B*T) + 2 * 1j * cf * numpy.pi*T) * T * (numpy.cos(2. * cf * numpy.pi*T) - numpy.sqrt(3. + pow(2,3/2)) * numpy.sin(2. * cf * numpy.pi*T))) * (-2. * numpy.exp( 4*1j*cf*numpy.pi*T) * T + 2. * numpy.exp(-(B*T) + 2*1j*cf*numpy.pi*T) * T * (numpy.cos(2. * cf * numpy.pi * T) + numpy.sqrt(3. + pow(2, 3/2)) * numpy.sin(2. * cf * numpy.pi*T))) / pow(-2. / numpy.exp(2.*B*T) - 2. * numpy.exp(4.*1j*cf*numpy.pi*T) + 2. * (1. + numpy.exp(4.*1j*cf*numpy.pi*T)) / numpy.exp(B*T), 4.))
    allfilts = numpy.ones(len(cf), float)
    fcoefs = numpy.concatenate(([A0*allfilts, ], [A11, ], [A12, ], [A13, ], [A14, ], [A2*allfilts, ], [B0*allfilts, ], [B1, ], [B2, ], [gain, ])).T
    #print "SHAPE", numpy.shape(fcoefs);

    return fcoefs


def chirpz(x, A, W, M):
    A = np.complex(A)
    W = np.complex(W)
    if np.issubdtype(np.complex, x.dtype) or np.issubdtype(np.float, x.dtype):
        dtype = x.dtype
    else:
        dtype = float
    x = np.asarray(x, dtype=np.complex)
    N = x.size
    L = int(2**np.ceil(np.log2(M + N-1)))

    n = np.arange(N, dtype=float)
    y = np.power(A, -n) * np.power(W, n**2 / 2.) * x
    Y = np.fft.fft(y, L)

    v = np.zeros(L, dtype=np.complex)
    v[:M] = np.power(W, -n[:M]**2/2.)
    v[L-N+1:] = np.power(W, -n[N-1:0:-1]**2/2.)
    V = np.fft.fft(v)
    g = np.fft.ifft(V*Y)[:M]
    k = np.arange(M)
    g *= np.power(W, k**2 / 2.)
    return g
