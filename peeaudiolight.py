#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# peeaudiolight.py
#
# Copyright (c) 2018 Geoffroy Peeters <geoffroy.peeters@ircam.fr>

# This file is part of ircamABCDJhardfeatures.

# ircamABCDJhardfeatures is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ircamABCDJhardfeatures is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with ircamABCDJhardfeatures.  If not, see <http://www.gnu.org/licenses/>.

# Author: G Peeters <geoffroy.peeters@ircam.fr.fr> http://www.ircam.fr

"""
    Light version of peeaudio.py -> target FitzGerald algorithm

:author: peeters@ircam.fr
:version: 1.0
:last-edit: 2018/04/30
"""


import sys, getopt
import numpy as np
#from scipy.fftpack import *
import scipy.io.wavfile
from scipy.signal import *
# median filtering
import scipy as sp

import os
from subprocess import Popen, PIPE
import time
import math
from scipy.ndimage import median_filter



class C_Descriptor():
    """
    Class for storing descriptor values
    Everything is considered here as descriptr (audio signal, STFT, Chromagram, SSM, ...)
    """

    verbose = False
    name = []
    data_v = []
    nbDim = []
    x_label = []
    y_label = []
    x_sr_hz = []
    y_sr_hz = []
    x_start = []
    y_start = []


    def __init__(self, name, data_v, x_label, x_sr_hz, x_start, y_label, y_sr_hz, y_start):
        """
        **Description:** initalization of an instance of a Descriptor object
        """

        # name of the content
        self.name = name
        # value of the content: can be a matrix (nbChannel=1, nbFrame) or a matrix (nbChannel=1, nbDim, nbFrame)
        self.data_v = data_v

        #self.nbDim = self.data_v.shape[0]
        #self.LT_n     = self.data_v.shape[1]

        # information for plotting
        # label to be displayed
        self.x_label = x_label
        self.y_label = y_label
        # sampling rate of the axes
        self.x_sr_hz = float(x_sr_hz)
        self.y_sr_hz = float(y_sr_hz)
        # starting point of the axes
        self.x_start = float(x_start)
        self.y_start = float(y_start)

        if False:
            if self.data_v.ndim == 1:
                print("1) %s sr:%f Dim1:%d" % (self.name, self.x_sr_hz, data_v.shape[0]))
            elif self.data_v.ndim == 2:
                print("2) %s sr:%f Dim1:%d Dim2:%d" % (self.name, self.x_sr_hz, data_v.shape[0], data_v.shape[1]))
                nbFrame = self.data_v.shape[1]
                T_v = self.x_start + np.arange(0., float(nbFrame)) / self.x_sr_hz
                print("\tstart:%f %f ... stop:%f" % (T_v[0], T_v[1], T_v[-1]))
                print("")
            elif self.data_v.ndim == 3:
                print("3) %s sr:%f Dim1:%d Dim2:%d Dim3:%d" % (self.name, self.x_sr_hz, data_v.shape[0], data_v.shape[1], data_v.shape[2]))
                nbFrame = self.data_v.shape[2]
                T_v = self.x_start + np.arange(0., float(nbFrame)) / self.x_sr_hz
                print("\tstart:%f %f ... stop:%f" % (T_v[0], T_v[1], T_v[-1]))
                print("")
            else:
                print("ERROR")


    def __setattr__(self, attrName, val):
        if hasattr(self, attrName):
            self.__dict__[attrName] = val
        else:
            raise Exception("self.%s note part of the fields" % attrName)


    def M_frameAnalysis(self, L_sec=0.08, STEP_sec=0.02, window_shape="hamming", remove_dc=True, mark_sec_v=np.zeros(0)):
        """
        **Description:** Split the data vector or matrix into frames and concatenate the frames into a matrix

        **Parameters:** L_sec, STEP_sec, window_shape, remove_dc

        **Status:** OK
        """

        #  --- self.data_v (nbChannel=1, LT_n)
        L_n = int(round(L_sec*self.x_sr_hz))
        if not(L_n % 2):
            L_n += 1  # in order to have an odd-length window

        LD_n = int((L_n-1)/2)
        L_sec = float(L_n)/self.x_sr_hz
        LD_sec = float(LD_n)/self.x_sr_hz

        if window_shape == "boxcar":
            fenetre_v = np.ones((L_n))
        elif window_shape == "hanning":
            fenetre_v = np.hanning(L_n)
        elif window_shape == "hamming":
            fenetre_v = np.hamming(L_n)
        elif window_shape == "blackman":
            fenetre_v = np.blackman(L_n)
        else:
            print("no other window_shape implemented so far")


        LT_n = self.data_v.shape[1]

        # --- convert analysis window to a matrix
        nbChannel = self.data_v.shape[0]
        # --- fenetre_v (nbChannel, L_n)
        fenetre_m = np.tile(fenetre_v, (nbChannel, 1))
        coefNorm = 0.5/np.sum(fenetre_v)

        if len(mark_sec_v):
            mark_n_v = (mark_sec_v-self.x_start)*self.x_sr_hz + 1
            STEP_sec = np.mean(np.diff(mark_sec_v))
        else:
            STEP_n = int(round(STEP_sec*self.x_sr_hz))
            STEP_sec = float(STEP_n)/self.x_sr_hz
            nbFrame = float(LT_n - L_n) / float(STEP_n)
            mark_n_v = (np.arange(0, nbFrame)*STEP_n)+LD_n
            mark_sec_v = self.x_start + (mark_n_v-1) / self.x_sr_hz

        nbFrame = len(mark_n_v)

        signal_3m = np.zeros((nbChannel, L_n, nbFrame))
        for numFrame in range(0, len(mark_n_v)):
            # --- signal_m (nbChannel, L_n)
            signal_m = self.data_v[:, int(mark_n_v[numFrame])-LD_n:int(mark_n_v[numFrame])+LD_n+1]
            # --- somme (nbChannel, 1)
            if remove_dc:
                somme_v = np.mean(signal_m, axis=1, keepdims=True)
            else:
                somme_v = 0

            signal_3m[:, :, numFrame] = coefNorm * np.multiply((signal_m-np.tile(somme_v, (1, L_n))), fenetre_m)


        # --- def __init__(self, name, data_v, x_label, x_sr_hz, x_start, y_label, y_sr_hz, y_start):
        output = C_Descriptor('M_frameAnalysis(' + self.name + ')', signal_3m, 'Frame [sec]', 1./STEP_sec, mark_sec_v[0], self.x_label, self.x_sr_hz, 0)
        return output








    # ===============================================
    def M_cplxFft(self, zp_factor=1):
        """
        """

        nbChannel = self.data_v.shape[0]
        L_n = self.data_v.shape[1]
        nbFrame = self.data_v.shape[2]
        N = zp_factor * F_nextPow2(L_n)

        fft_3m = np.zeros((self.data_v.shape[0], N/2+1, nbFrame), 'complex64')

        for numFrame in range(0, nbFrame):
            fft_3m[:, :, numFrame] = np.fft.rfft(self.data_v[:, :, numFrame], int(N))

        output = C_Descriptor('M_cplxFft(' + self.name + ')', fft_3m, 'Frame [sec]', self.x_sr_hz, self.x_start, 'Frequency', 1.0*N/self.y_sr_hz, 0.)
        return output




    # ===============================================
    def M_fitzGerald(self, L_sec=0.08, STEP_sec=0.02):
        """
        """

        # === PARAM ===
        do_mask_hard1_soft2 = 1
        param_p = 2
        param_Lfilter_sec = 0.2

        Cw = 2.35
        Bw = Cw/(L_sec)
        param_Lfilter_hz = 12*(Bw)
        param_beta = 1.5    # === Driegder
        # === PARAM ===

        nbChannel = self.data_v.shape[0]
        NN = self.data_v.shape[1]
        nbFrame = self.data_v.shape[2]

        amfft_3m = np.abs(self.data_v)
        H_amfft_3m = np.zeros((nbChannel, NN, nbFrame))
        P_amfft_3m = np.zeros((nbChannel, NN, nbFrame))

        Lfilter_n = np.ceil(param_Lfilter_sec*1./STEP_sec)
        if not(Lfilter_n % 2):
            Lfilter_n += 1
        N = (NN-1)*2
        # --- audiosr_hz = 1/(self.y_sr_hz/N) = 44100 Hz
        STEP_hz = 1/(self.y_sr_hz/N) / N
        Lfilter_k = np.ceil(param_Lfilter_hz/STEP_hz)
        if not(Lfilter_k % 2):
            Lfilter_k += 1

        for numChannel in range(0, nbChannel):
            H_amfft_3m[numChannel, :, :] = median_filter(amfft_3m[numChannel, :, :], size=(1, int(Lfilter_n)), mode='reflect')
            P_amfft_3m[numChannel, :, :] = median_filter(amfft_3m[numChannel, :, :], size=(int(Lfilter_k), 1), mode='reflect')

        if do_mask_hard1_soft2 == 1:
            # === Hard Mark
            MaskH_3m = np.zeros((nbChannel, NN, nbFrame))
            posH = np.where(H_amfft_3m > (param_beta*P_amfft_3m))
            MaskH_3m[posH] = 1

            MaskP_3m = np.zeros((nbChannel, NN, nbFrame))
            posP = np.where(P_amfft_3m >= (param_beta*H_amfft_3m))
            MaskP_3m[posP] = 1
        else:
            # === Soft Mask
            MaskH_3m = np.divide(H_amfft_3m**param_p, (H_amfft_3m**param_p + P_amfft_3m**param_p))
            MaskP_3m = np.divide(P_amfft_3m**param_p, (H_amfft_3m**param_p + P_amfft_3m**param_p))

        MaskR_3m = 1 - (MaskH_3m + MaskP_3m)

        # --- mean over Channel and Frequency
        enerTotal_v = np.mean(np.mean(amfft_3m , axis=0), axis=0)
        enerHarmo_v = np.mean(np.mean(np.multiply(amfft_3m, MaskH_3m), axis=0), axis=0)
        enerPercu_v = np.mean(np.mean(np.multiply(amfft_3m, MaskP_3m), axis=0), axis=0)
        enerResi_v = np.mean(np.mean(np.multiply(amfft_3m, MaskR_3m), axis=0), axis=0)

        pos_v = np.where(enerTotal_v > 0.0)
        C1 = np.mean(np.divide(enerHarmo_v[pos_v], enerTotal_v[pos_v]), axis=0)
        C2 = np.mean(np.divide(enerPercu_v[pos_v], enerTotal_v[pos_v]), axis=0)
        C3 = np.mean(np.divide(enerResi_v[pos_v], enerTotal_v[pos_v]), axis=0)

        return C1, C2, C3


class C_AudioAnalysis(C_Descriptor):
    """
    **Description:** Specific class for loading audio files from .mp3 or .wav

    **Parameters:

    **Status:** OK
    """

    def __init__(self, audioFile, do_stereo2mono=False):

        assert(os.path.isfile(audioFile)), "file '%s' does not exist" % (audioFile)
        assert(audioFile.endswith('wav')), "file '%s' must be a .wav file" % (audioFile)

        sr_hz, data_v = scipy.io.wavfile.read(audioFile)

        # --- 2018/07/09: to speed up computation: only consider first 30sec
        #data_v = data_v[:10*sr_hz, :]

        # --- data_v (LT_n, nbChannel)
        data_v = data_v / 32767.0

        # --- data_v (length, nbChannel)
        # --- ou
        # --- data_v (length, )
        if data_v.ndim == 1:
            data_v = np.reshape(data_v, (len(data_v), 1))
        else:
            if do_stereo2mono:
                # --- reduce stereo -> mono
                data_v = np.mean(data_v, axis=1)
                data_v = np.reshape(data_v, (len(data_v), 1))

        # --- convert to data_v (nbChannel, LT_n)
        data_v = data_v.T


        # --- C_Descriptor.__init__(self, audioFile, data_v, 'Time [sec]', sr_hz, 0., 'Audio-value', 1., 0.)
        C_Descriptor.__init__(self, 'audio', data_v, 'Time [sec]', sr_hz, 0., 'Audio-value', 1., 0.)


def F_nextPow2(i):
    n = 2
    while n < i:
        n = n * 2
    return n
