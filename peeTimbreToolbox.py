# -*- coding: utf-8 -*-
#
# timbre_descriptor.py
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

# TODO: virer les warning
# TODO: lecture mp3, aac, ...
# TODO: plug avec main TU-Berlin


import numpy as np
from collections import namedtuple
import scipy
import scipy.signal
from scipy.io import wavfile

import swipep as swp                # used for single-F0 estimation
import my_tools as mt

import sys
import getopt
import time
#import ipdb

import math # for trunc

#import matplotlib.pyplot as plt


EPS = mt.EPS
NB_DESC = 164
desc_settings = namedtuple("desc_settings", "b_AS b_TEE b_TEE_global b_STFTmag b_STFTpow b_Harmonic b_ERBfft b_ERBgam xcorr_nb_coeff threshold_harmo nb_harmo")
dPart = namedtuple("dPart", "f_Freq_v f_Ampl_v");
config_s = desc_settings(
# descriptors from the Audio Signal
b_AS = 0,
# descriptors from the Temporal Energy Envelope
b_TEE = 1,
# compute Log-Attack-Time, ...
b_TEE_global = 1,
# descriptors from the STFT magnitude
b_STFTmag = 1,
# descriptors from the STFT power
b_STFTpow = 1,
# descriptors from Harmonic Sinusoidal Modeling representation
b_Harmonic = 1,
# descriptors from ERB representation (ERB being computed using FFT)
b_ERBfft = 1,
# descriptors from ERB representation (ERB being computed using Gamma Tone Filter)
b_ERBgam = 1,
# === defines the number of auto-correlation coefficients that will be sued
xcorr_nb_coeff = 12,
# === defines the threshold [0,1] below which harmonic-features are not computed
threshold_harmo = 0.3,
# === defines the number of harmonics that will be extracted
nb_harmo = 20)

nbits = 16
MAX_VAL = pow(2, (nbits-1)) * 1.0



####################################################################################
####                          TEST SCRIPTS                                      ####
####################################################################################
## Example of test script
# import timbre_descriptor, numpy, scipy, my_tools as mt
# s, Fs, desc, param_val, field_name = timbre_descriptor.test_descriptor()
# s, Fs, desc_struc, t = timbre_descriptor.test_descriptor2()


def check_values(param_val, param_ref, field_name, Thr=0.1):

    err = np.abs(param_ref - param_val)
    h_err = err / abs(param_ref)
    h_err[h_err > 1.] = 1.    ## normalized error

    i_pbm = (h_err > Thr).nonzero()[0]
    print("/!\\",len(i_pbm), "erreurs detectees (seuil",Thr,"):\n")

    for i in i_pbm:
        print(i,") ", field_name[i]," ",param_val[i], " != ", param_ref[i]," (err=",h_err[i],")")
    return h_err


def F_save(descHub_d, filename):
    import pickle
    with open(filename, 'wb') as output:
        pickle.dump(descHub_d, output, pickle.HIGHEST_PROTOCOL)

    #import json
    #with open(filename, 'w') as outfile: json.dump(descHub_d, outfile)
    return


def F_load(filename):
    import pickle
    with open(filename, 'r') as output:
        descHub_d = pickle.load(output)

    #import json
    #with open(filename, 'w') as outfile: json.dump(descHub_d, outfile)
    return descHub_d


####################################################################################
####                          Main functions                                    ####
####################################################################################

#  Compute statistics from time series (median / iqr)
#  name: unknown
#  @param
#  @return


def F_temporalModeling(descHub_d):

    for key1 in descHub_d.keys():
        for key2 in descHub_d[key1].keys():
            if ((descHub_d[key1][key2]['value'].shape[0] == 1) & (descHub_d[key1][key2]['value'].shape[1] > 1)):

                descHub_d[key1][key2]['min'] = np.amin(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['max'] = np.amax(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['mean'] = np.mean(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['std'] = np.std(descHub_d[key1][key2]['value'], axis=1)

                descHub_d[key1][key2]['median'] = np.median(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['iqr'] = 0.7413 * (np.percentile(descHub_d[key1][key2]['value'], 75, axis=1) - np.percentile(descHub_d[key1][key2]['value'], 25, axis=1))
                # --- crest= max / mean
                descHub_d[key1][key2]['crest'] = np.max(descHub_d[key1][key2]['value'], axis=1) / np.mean(descHub_d[key1][key2]['value'], axis=1)


            elif ((descHub_d[key1][key2]['value'].shape[0] > 1) & (descHub_d[key1][key2]['value'].shape[1] > 1)):

                descHub_d[key1][key2]['min'] = np.amin(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['max'] = np.amax(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['mean'] = np.mean(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['std'] = np.std(descHub_d[key1][key2]['value'], axis=1)

                descHub_d[key1][key2]['median'] = np.median(descHub_d[key1][key2]['value'], axis=1)
                descHub_d[key1][key2]['iqr'] = 0.7413 * (np.percentile(descHub_d[key1][key2]['value'], 75, axis=1) - np.percentile(descHub_d[key1][key2]['value'], 25, axis=1))
                # --- crest= max / mean
                descHub_d[key1][key2]['crest'] = np.divide(np.max(descHub_d[key1][key2]['value'], axis=1), np.mean(descHub_d[key1][key2]['value'], axis=1))

            else:

                descHub_d[key1][key2]['min'] = descHub_d[key1][key2]['value']
                descHub_d[key1][key2]['max'] = descHub_d[key1][key2]['value']
                descHub_d[key1][key2]['mean'] = descHub_d[key1][key2]['value']
                descHub_d[key1][key2]['std'] = 0.0

                descHub_d[key1][key2]['median'] = descHub_d[key1][key2]['value']
                descHub_d[key1][key2]['iqr'] = 0.0


            if False:
                # -------------------------------------------
                # -------------------------------------------
                # -------------------------------------------
                nbFrame = descHub_d[key1][key2]['value'].shape[1]
                if nbFrame>1:
                    for numDim in range(0,descHub_d[key1][key2]['value'].shape[0]):
                        plt.figure(1)
                        plt.clf()
                        plt.subplot(111)
                        plt.plot(range(0,nbFrame), descHub_d[key1][key2]['value'][numDim,:])
                        m_v = np.ones((nbFrame))*descHub_d[key1][key2]['median'][numDim]
                        s_v = np.ones((nbFrame))*descHub_d[key1][key2]['iqr'][numDim]
                        plt.plot(range(0,nbFrame), m_v, 'g')
                        plt.plot(range(0,nbFrame), m_v-s_v, 'r')
                        plt.plot(range(0,nbFrame), m_v+s_v, 'r')
                        plt.title("%s/%s/%d" % (key1, key2,numDim))
                        plt.grid(True)
                        plt.show(block=False)
                        plt.draw()
                        #raw_input()
                else:
                    print("%s/%s: %f" % (key1, key2, descHub_d[key1][key2]['value']))

    return descHub_d



def F_computeAllDescriptor(audio_v, sr_hz):
    """
        Main Function : compute all descriptor for given signal trame_s
        name: unknown
        @param
        @return
    """

    descHub_d = {}

    if np.isscalar(sr_hz):
        sr_hz = np.array([sr_hz])

    print(audio_v.shape)
    print(sr_hz)

    audio_v = audio_v / (np.max(audio_v) + EPS);  # normalize input sound
    #ret_desc    = np.zeros((1, NB_DESC), float);

    descHub_d = {}

    import time

    """ 1) descriptors from the Temporal Energy Envelope (do_s.b_TEE=1)  [OK] """
    if config_s.b_AS:
        t = time.time()
        descHub_d['AS'] = F_computeDescriptorSignal(audio_v, sr_hz)
        print("F_computeDescriptorSignal\t%f" % (time.time() - t))

    if config_s.b_TEE:
        t = time.time()
        descHub_d['TEE'] = F_computeDescriptorEnv(audio_v, sr_hz)
        print("F_computeDescriptorEnv\t%f" % (time.time() - t))

    """ 2) descriptors from the STFT magnitude and STFT power (do_s.b_STFTmag= 1) """
    if (config_s.b_STFTmag or config_s.b_STFTpow):
        t = time.time()
        S_mag, S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v = F_representationFft(audio_v, sr_hz)
        print("F_representationFft\t%f" % (time.time() - t))

    if config_s.b_STFTmag:
        t = time.time()
        descHub_d['STFTmag'] = F_computeDescriptorSpectrum(S_mag, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v)
        print("F_computeDescriptorSpectrum\t%f" % (time.time() - t))

    if config_s.b_STFTpow:
        t = time.time()
        descHub_d['STFTpow'] = F_computeDescriptorSpectrum(S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v)
        print("F_computeDescriptorSpectrum\t%f" % (time.time() - t))

    """ 3) descriptors from Harmonic Sinusoidal Modeling representation (do_s.b_Harmonic=1) """
    if config_s.b_Harmonic:
        t = time.time()
        f0_hz_v, f_DistrPts_m, PartTrax_s = F_representationHarmonic(audio_v, sr_hz)
        print("F_representationHarmonic\t%f" % (time.time() - t))

        t = time.time()
        descHub_d['Harmonic'] = F_computeDescriptorHarmonic(f0_hz_v, f_DistrPts_m, PartTrax_s)
        print("F_computeDescriptorHarmonic\t%f" % (time.time() - t))

    #m = scipy.io.loadmat('sig_erb.mat');
    #trame_s = np.squeeze(m['f_Sig_v']);

    """ 4) descriptors from ERB representation (ERB being computed using FFT) """
    if (config_s.b_ERBfft or config_s.b_ERBgam):
        t = time.time()
        S_erb, S_gam, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2 = F_representationERB(audio_v, sr_hz)
        print("F_representationERB\t%f" % (time.time() - t))

    if config_s.b_ERBfft:
        t = time.time()
        descHub_d['ERBfft'] = F_computeDescriptorSpectrum(S_erb, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1)
        print("F_computeDescriptorSpectrum\t%f" % (time.time() - t))

    if config_s.b_ERBgam:
        t = time.time()
        descHub_d['ERBgam'] = F_computeDescriptorSpectrum(S_gam, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2)
        print("F_computeDescriptorSpectrum\t%f" % (time.time() - t))

    return descHub_d





def F_computeDescriptorEnv(audio_v, sr_hz, do_Global=True):
    """
        Compute Temporal escriptors from input trame_s signal
        name: time_desc(trame_s, Fs)
        @param
        @return
    """
    #print "F_computeDescriptorEnv"

    sr_hz = float(sr_hz)
    Fc = 5.0
    f_ThreshNoise = 0.15
    #trame_s = trame_s / (np.max(trame_s) + EPS);  # normalize input sound

    """ horrible hack to avoid big prime factors in FFT """
    hacked_size = math.trunc(np.ceil(audio_v.size / sr_hz) * sr_hz)
    if hacked_size > audio_v.size :
      audio_v = np.append(audio_v, np.zeros(hacked_size - audio_v.size))

    """ compute signal enveloppe (Ok) """
    f_AnaSig_v = scipy.signal.hilbert(audio_v)    # analytic signal
    f_AmpMod_v = abs(f_AnaSig_v)                  # amplitude modulation of analytic signal

    ## seems to have problem with Python (replaced with Matlab version)...
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    [B_v, A_v] = scipy.signal.butter(3., 2. * Fc / sr_hz, 'lowpass', False, 'ba')          #3rd order Butterworth filter
    #B_v = [ 4.51579830e-11];
    #A_v = [ 1., -2.99857524, 2.9971515, -0.99857626];

    #m = scipy.io.loadmat('butter3.mat');
    #B_v = np.squeeze(m['B_v']);
    #A_v = np.squeeze(m['A_v']);

    f_Env_v = scipy.signal.lfilter(B_v, A_v, np.squeeze(np.array(f_AmpMod_v)));

    if config_s.b_TEE_global:
        # Log-Attack (Ok)
        f_LAT, f_Incr, f_Decr, f_ADSR_v = F_computeLogAttack(f_Env_v, sr_hz, f_ThreshNoise);

        # temporal centroid (in seconds) (Ok)
        f_TempCent = F_computeTempCentroid(f_Env_v, f_ThreshNoise) / sr_hz;
        # === Effective duration (Ok)
        f_EffDur = F_computeEffectiveDuration(f_Env_v, 0.4) / sr_hz;    # effective duration (in seconds)
        # === Energy modulation (tremolo) (Ok)
        f_FreqMod, f_AmpMod = F_computeModulation(f_Env_v, f_ADSR_v, sr_hz);

    descHub_d = {}

    if config_s.b_TEE_global:
        descHub_d['Att'] = {'value': np.ones((1,1))*f_ADSR_v[0]}
        descHub_d['Dec'] = {'value': np.ones((1,1))*f_ADSR_v[1]}
        descHub_d['Rel'] = {'value': np.ones((1,1))*f_ADSR_v[4]}
        descHub_d['LAT'] = {'value': np.ones((1,1))*f_LAT}            # === log attack time
        descHub_d['AttSlope'] = {'value': np.ones((1,1))*f_Incr}            # === temporal increase
        descHub_d['DecSlope'] = {'value': np.ones((1,1))*f_Decr}            # === temporal decrease
        descHub_d['TempCent'] = {'value': np.ones((1,1))*f_TempCent}        # === temporal centroid
        descHub_d['EffDur'] = {'value': np.ones((1,1))*f_EffDur}        # === effective duration
        descHub_d['FreqMod'] = {'value': np.ones((1,1))*f_FreqMod}        # === energy modulation frequency
        descHub_d['AmpMod'] = {'value': np.ones((1,1))*f_AmpMod}        # === energy modulation amplitude

    descHub_d['RMSEnv'] = {'value': f_Env_v.reshape(1,f_Env_v.shape[0])}        # === GFP 2010/11/16

    return descHub_d



def F_computeDescriptorSignal(audio_v, sr_hz):
    """
        Compute Temporal escriptors from input trame_s signal
        name: time_desc(trame_s, Fs)
        @param
        @return
    """
    #print "F_computeDescriptorSignal"

    f_hopSize_sec = 128.0/44100         # === is 0.0029s at 44100Hz
    f_winLen_sec = 1024.0/44100         # === is 0.0232s at 44100Hz
    hopSize_n = int(round(f_hopSize_sec * sr_hz))
    winLen_n = int(round(f_winLen_sec * sr_hz))
    win_v = scipy.signal.hamming(winLen_n)
    LT_n = len(audio_v)
    i2 = np.arange(0, winLen_n, 1)
    nbFrame = int((LT_n-winLen_n) / hopSize_n) + 1
    f_AutoCoeffs_v = np.zeros((config_s.xcorr_nb_coeff, nbFrame), float)
    f_ZcrRate_v = np.zeros((1, nbFrame), float)
    frameInd_v = np.arange(0, winLen_n, 1)

    idx = 0
    for n in range(0, LT_n-winLen_n, hopSize_n):
        #print "Processing frame ", (idx+1)," / ", (nb_trame),"\n"
        i1_v = np.round(int(n) + frameInd_v)
        f_Frm_v = audio_v[i1_v] * win_v

        """ Autocorrelation """
        f_Coeffs_v = np.fft.fftshift(mt.xcorr(f_Frm_v + EPS))    # GFP divide by zero issue
        f_AutoCoeffs_v[:, idx] = f_Coeffs_v[0:(config_s.xcorr_nb_coeff)]        # only save 12 coefficients

        """ Zero crossing rate """
        i_Sign_v = np.sign(f_Frm_v - np.mean(f_Frm_v))
        i_Zcr_v = np.diff(i_Sign_v).nonzero()[0]
        f_ZcrRate_v[0, idx] = len(i_Zcr_v) / (len(f_Frm_v) / sr_hz)
        idx = idx + 1

    descHub_d = {}
    descHub_d['AutoCorr'] = {'value': f_AutoCoeffs_v}
    descHub_d['ZcrRate'] = {'value': f_ZcrRate_v}

    #.(sprintf('AutoCorr%d',num_dim)) = f_AutoCoeffs_v(num_dim,:);    # === autocorrelation
    return descHub_d



def F_computeDescriptorSpectrum( f_DistrPts_m, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v ):
    """
        Compute descriptor from spectral representation (FFT/ERB/GAM)
        name: F_computeDescriptor
        @param
        @return
    """
    #print "F_computeDescriptorSpectrum"

    # --- f_DistrPts_m (i_SizeY=N, i_SizeX=nbFrame)

    #i_SizeY, i_SizeX    = f_DistrPts_m.shape;
    x_tmp = sum(f_DistrPts_m, 0) + EPS
    f_ProbDistrY_m = f_DistrPts_m / np.repeat( [x_tmp,], i_SizeY, 0)    # === normalize distribution in Y dim
    i_NumMoments = 4                                                # === Number of moments to compute
    f_Moments_m = np.zeros((i_NumMoments, i_SizeX), float)        # === create empty output array for moments

    """ Calculate moments """
    # === f_Moments_m must be empty on first iter.
    f_MeanCntr_m = np.repeat( np.array([f_SupY_v,]).T, i_SizeX, 1) - np.repeat( np.array([f_Moments_m[0,:],]), i_SizeY, 0)

    for i in range(0, i_NumMoments):
        f_Moments_m[i, :] = sum(pow(f_MeanCntr_m, float(i+1)) * f_ProbDistrY_m)

    """ Descriptors from first 4 moments """
    f_Centroid_v = f_Moments_m[0, :]
    f_StdDev_v = np.sqrt(f_Moments_m[1, :])
    f_Skew_v = f_Moments_m[2, :] / pow(f_StdDev_v+EPS, 3.)
    f_Kurtosis_v = f_Moments_m[3, :] / pow(f_StdDev_v+EPS, 4.)

    """ Spectral slope (linear regression) """
    f_Num_v = i_SizeY * (f_SupY_v.dot(f_ProbDistrY_m)) - np.sum(f_SupY_v) * sum(f_ProbDistrY_m)
    f_Den = i_SizeY * sum(f_SupY_v ** 2.) - pow(sum(f_SupY_v), 2.)
    f_Slope_v = f_Num_v / (EPS + f_Den)

    """ Spectral decrease (according to peeters report) """
    f_Num_m = f_DistrPts_m[1:i_SizeY, :] - np.repeat( [f_DistrPts_m[0,:] ,], i_SizeY-1, 0)
    ## a verifier
    #print sum(f_DistrPts_m[0,:])
    #my_plot(f_DistrPts_m[0,:])
    f_Den_v = 1. / np.arange(1, i_SizeY, 1.)
    f_SpecDecr_v = np.dot(f_Den_v, f_Num_m) / np.sum(f_DistrPts_m+EPS, axis=0); #[1:i_SizeY,:]

    #print  "chatt: ", np.shape(f_Num_m), " - ", np.shape(f_Den_v)
    #print "SUM:", np.sum(f_DistrPts_m[1:i_SizeY, :])
    #print "SUM:", np.sum(np.repeat([f_DistrPts_m[0,:],], i_SizeY-1, 0))
    #my_plot(f_DistrPts_m[0,:])

    """ Spectral roll-off """
    f_Thresh = 0.95
    f_CumSum_m = np.cumsum(f_DistrPts_m, axis=0)
    f_Sum_v = f_Thresh * np.sum(f_DistrPts_m, axis=0)
    i_Bin_m = f_CumSum_m > np.repeat( [f_Sum_v,], i_SizeY, 0 )
    tmp = np.cumsum(i_Bin_m, axis=0)
    trash, i_Ind_v = ( tmp.T == 1 ).nonzero()
    f_SpecRollOff_v = f_SupY_v[i_Ind_v]

    """ Spectral variation (Spect. Flux) """
    f_CrossProd_v = np.sum( f_DistrPts_m * np.concatenate( (np.zeros((1, i_SizeY), float), f_DistrPts_m[:,0:(i_SizeX-1)].T ) ).T , axis=0)
    f_AutoProd_v = np.sum( pow(f_DistrPts_m, 2.), axis=0 ) * np.sum( pow( np.concatenate( (np.zeros((1,i_SizeY), float), f_DistrPts_m[:,0:(i_SizeX-1)].T)).T , 2. ) , axis=0)

    f_SpecVar_v = 1. - f_CrossProd_v / (np.sqrt(f_AutoProd_v) + EPS)
    f_SpecVar_v[0] = f_SpecVar_v[1]    # === the first value is alway incorrect because of "c.f_DistrPts_m .* [zeros(c.i_SizeY,1)"

    """ Energy """
    f_Energy_v = np.sum(f_DistrPts_m, axis=0)

    """ Spectral Flatness """
    f_GeoMean_v = np.exp( (1. / i_SizeY) * np.sum(np.log( f_DistrPts_m+EPS ), axis=0) )
    f_ArthMean_v = np.sum(f_DistrPts_m, axis=0) / float(i_SizeY)
    f_SpecFlat_v = f_GeoMean_v / (f_ArthMean_v+EPS)

    """ Spectral Crest Measure """
    f_SpecCrest_v = np.max(f_DistrPts_m, axis=0) / (f_ArthMean_v + EPS)

    # ==============================
    # ||| Build output structure |||
    # ==============================
    descHub_d = {}
    descHub_d['SpecCent'] = {'value': f_Centroid_v.reshape(1, f_Centroid_v.shape[0])}    # spectral centroid - OK
    descHub_d['SpecSpread'] = {'value': f_StdDev_v.reshape(1, f_StdDev_v.shape[0])}    # spectral standard deviation - OK
    descHub_d['SpecSkew'] = {'value': f_Skew_v.reshape(1, f_Skew_v.shape[0])}        # spectral skew - OK
    descHub_d['SpecKurt'] = {'value': f_Kurtosis_v.reshape(1, f_Kurtosis_v.shape[0])}    # spectral kurtosis - OK
    descHub_d['SpecSlope'] = {'value': f_Slope_v.reshape(1, f_Slope_v.shape[0])}        # spectral slope - OK
    descHub_d['SpecDecr'] = {'value': f_SpecDecr_v.reshape(1, f_SpecDecr_v.shape[0])}    # spectral decrease - ?
    descHub_d['SpecRollOff'] = {'value': f_SpecRollOff_v.reshape(1, f_SpecRollOff_v.shape[0])}# spectral roll-off  - OK
    descHub_d['SpecVar'] = {'value': f_SpecVar_v.reshape(1, f_SpecVar_v.shape[0])}    # spectral variation - OK
    descHub_d['FrameErg'] = {'value': f_Energy_v.reshape(1, f_Energy_v.shape[0])}    # frame energy - OK

    descHub_d['SpecFlat'] = {'value': f_SpecFlat_v.reshape(1, f_SpecFlat_v.shape[0])}    # spectral flatness - OK
    descHub_d['SpecCrest'] = {'value': f_SpecCrest_v.reshape(1, f_SpecCrest_v.shape[0])}    # spectral crest - OK

    return descHub_d



def F_computeDescriptorHarmonic(f_F0_v, f_DistrPts_m, PartTrax_s):
    """
        Compute Harmonic descriptors
        name: unknown
        @param
        @return
    """

    #print "F_computeDescriptorHarmonic"

    i_Offset = 0
    i_EndFrm = len(PartTrax_s)

    DEFAULT_VALUE = -999
    descHub_d = {}
    if i_EndFrm == 0:
        descHub_d['HarmErg'] = {'value': DEFAULT_VALUE}
        descHub_d['NoiseErg'] = {'value': DEFAULT_VALUE}
        descHub_d['Noisiness'] = {'value': DEFAULT_VALUE}
        descHub_d['F0'] = {'value': DEFAULT_VALUE}
        descHub_d['InHarm'] = {'value': DEFAULT_VALUE}
        descHub_d['TriStim1'] = {'value': DEFAULT_VALUE}
        descHub_d['TriStim2'] = {'value': DEFAULT_VALUE}
        descHub_d['TriStim3'] = {'value': DEFAULT_VALUE}
        descHub_d['HarmDev'] = {'value': DEFAULT_VALUE}
        descHub_d['OddEveRatio'] = {'value': DEFAULT_VALUE}

        descHub_d['SpecCent'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecSpread'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecSkew'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecKurt'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecSlope'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecDecr'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecRollOff'] = {'value': DEFAULT_VALUE}
        descHub_d['SpecVar'] = {'value': DEFAULT_VALUE}
        descHub_d['FrameErg'] = {'value': DEFAULT_VALUE}

        return descHub_d
    else:
        descHub_d['HarmErg'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['NoiseErg'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['Noisiness'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['F0'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['InHarm'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['TriStim1'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['TriStim2'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['TriStim3'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['HarmDev'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['OddEveRatio'] = {'value': np.zeros((1, i_EndFrm-1), float)}

        descHub_d['SpecCent'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecSpread'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecSkew'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecKurt'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecSlope'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecDecr'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecRollOff'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['SpecVar'] = {'value': np.zeros((1, i_EndFrm-1), float)}
        descHub_d['FrameErg'] = {'value': np.zeros((1, i_EndFrm-1), float)}

    for i in range(1, i_EndFrm):
        """ Energy """
        f_Energy = sum(f_DistrPts_m[:, i+i_Offset])
        f_HarmErg = sum(pow(PartTrax_s[i, 1, :], 2.)) #Amp
        f_NoiseErg = f_Energy - f_HarmErg

        """ Noisiness """
        f_Noisiness = f_NoiseErg / (f_Energy+EPS)

        """ Inharmonicity """
        i_NumHarm = len(PartTrax_s[i, 1, :])
        if (i_NumHarm < 5):
            f_InHarm = []
            continue

        f_Harms_v = f_F0_v[i] * np.arange(1, i_NumHarm+1, 1.)
        f_InHarm = np.sum(abs(PartTrax_s[i, 0, :] - f_Harms_v) * pow(PartTrax_s[i, 1, :], 2.) ) / (np.sum(pow(PartTrax_s[i, 1, :], 2.)) + EPS) * 2. / f_F0_v[i]

        """ Harmonic spectral deviation """
        f_SpecEnv_v = np.zeros(i_NumHarm, float)
        f_SpecEnv_v[0] = PartTrax_s[i, 1, 0]
        l = len(PartTrax_s[i,1, :])
        f_SpecEnv_v[1:(i_NumHarm-1)] = (PartTrax_s[i, 1, 0:(l-2)] + PartTrax_s[i, 1, 1:(l-1)] + PartTrax_s[i, 1, 2:] ) / 3.
        f_SpecEnv_v[i_NumHarm-1] = (PartTrax_s[i, 1, l-2] + PartTrax_s[i, 1, l-1] ) / 2.
        f_HarmDev = np.sum(abs(PartTrax_s[i, 1, :] - f_SpecEnv_v)) / i_NumHarm

        """ Odd to even harmonic ratio """
        f_OddEvenRatio = np.sum(pow(PartTrax_s[i, 1, 0::2], 2.)) / (np.sum(pow(PartTrax_s[i, 1, 1::2], 2.)) + EPS)

        """ Harmonic tristimulus """
        f_TriStim_v = np.zeros(3, float)
        f_TriStim_v[0] = PartTrax_s[i, 1, 0]  / (sum(PartTrax_s[i, 1, :]) + EPS)
        f_TriStim_v[1] = sum(PartTrax_s[i, 1, 1:4]) / (sum(PartTrax_s[i, 1, :]) + EPS)
        f_TriStim_v[2] = sum(PartTrax_s[i, 1, 4:]) / (sum(PartTrax_s[i, 1, :]) + EPS)

        """ Harmonic centroid """
        f_NormAmpl_v = PartTrax_s[i, 1, :] / (sum(PartTrax_s[i, 1, :] ) + EPS)
        f_Centroid = sum(PartTrax_s[i, 0, :] * f_NormAmpl_v)
        f_MeanCentrFreq = PartTrax_s[i, 0, :] - f_Centroid

        """ Harmonic spread """
        f_StdDev = np.sqrt(sum(pow(f_MeanCentrFreq, 2.) * f_NormAmpl_v))

        """ Harmonic skew """
        f_Skew = sum(pow(f_MeanCentrFreq, 3.) * f_NormAmpl_v) / pow(f_StdDev + EPS, 3.)

        """ Harmonic kurtosis """
        f_Kurtosis = sum(pow(f_MeanCentrFreq, 4.) * f_NormAmpl_v) / pow(f_StdDev + EPS, 4.)

        """ Harmonic spectral slope (linear regression) """
        f_Num = i_NumHarm * np.sum(PartTrax_s[i, 0, :] * f_NormAmpl_v) - np.sum(PartTrax_s[i, 0, :])
        f_Den = i_NumHarm * np.sum(pow(PartTrax_s[i, 0, :], 2.)) - np.sum(pow(PartTrax_s[i, 0, :], 2.))
        f_Slope = f_Num / f_Den

        """ Spectral decrease (according to peeters report) """
        f_Num = sum((PartTrax_s[i, 1, 1:i_NumHarm] - PartTrax_s[i, 1, 0]) / np.arange(1., i_NumHarm))
        f_Den = sum(PartTrax_s[i, 1, 1:i_NumHarm])
        f_SpecDecr = f_Num / (f_Den+EPS)

        """ Spectral roll-off """
        f_Thresh = 0.95
        f_CumSum_v = np.cumsum( PartTrax_s[i, 1, :])
        f_CumSumNorm_v = f_CumSum_v / (sum(PartTrax_s[i, 1, :]) + EPS)
        i_Pos = (f_CumSumNorm_v > f_Thresh).nonzero()[0]
        if len(i_Pos) > 0:
            f_SpecRollOff = PartTrax_s[i, 0, i_Pos[0]]
        else:
            f_SpecRollOff = PartTrax_s[i, 0, 0]

        """ Spectral variation (Spect. Flux) """
        # === Insure that prev. frame has same size as current frame by zero-padding
        i_Sz = max(len(PartTrax_s[i-1, 1, :]), len(PartTrax_s[i, 1, :]))
        f_PrevFrm_v = np.zeros(i_Sz, float)
        f_CurFrm_v = np.zeros(i_Sz, float)

        f_PrevFrm_v[0:len(PartTrax_s[i-1, 1, :])] = PartTrax_s[i-1, 1, :]
        f_CurFrm_v[0:len(PartTrax_s[i, 1, :])] = PartTrax_s[i, 1, :]

        f_CrossProd = sum(f_PrevFrm_v * f_CurFrm_v)
        f_AutoProd = np.sqrt(sum(pow(f_PrevFrm_v, 2.)) * sum(pow(f_CurFrm_v, 2.)))
        f_SpecVar = 1 - f_CrossProd / (f_AutoProd+EPS)

        """ Build output structure """
        descHub_d['HarmErg']['value'][0, i-1] = f_HarmErg
        descHub_d['NoiseErg']['value'][0, i-1] = f_NoiseErg
        descHub_d['Noisiness']['value'][0, i-1] = f_Noisiness
        descHub_d['F0']['value'][0, i-1] = f_F0_v[i]
        descHub_d['InHarm']['value'][0, i-1] = f_InHarm
        descHub_d['TriStim1']['value'][0, i-1] = f_TriStim_v[0]
        descHub_d['TriStim2']['value'][0, i-1] = f_TriStim_v[1]
        descHub_d['TriStim3']['value'][0, i-1] = f_TriStim_v[2]
        descHub_d['HarmDev']['value'][0, i-1] = f_HarmDev
        descHub_d['OddEveRatio']['value'][0, i-1] = f_OddEvenRatio

        descHub_d['SpecCent']['value'][0, i-1] = f_Centroid
        descHub_d['SpecSpread']['value'][0, i-1] = f_StdDev
        descHub_d['SpecSkew']['value'][0, i-1] = f_Skew
        descHub_d['SpecKurt']['value'][0, i-1] = f_Kurtosis
        descHub_d['SpecSlope']['value'][0, i-1] = f_Slope
        descHub_d['SpecDecr']['value'][0, i-1] = f_SpecDecr
        descHub_d['SpecRollOff']['value'][0, i-1] = f_SpecRollOff
        descHub_d['SpecVar']['value'][0, i-1] = f_SpecVar
        descHub_d['FrameErg']['value'][0, i-1] = f_Energy

    return descHub_d


def F_representationFft(audio_v, sr_hz):
    """
    """

    #print "F_representationFft"

    #i_FFTSize    = 2048;
    f_WinSize_sec = 1025./44100.
    f_HopSize_sec = 256./44100.

    i_WinSize = int(f_WinSize_sec*sr_hz)
    i_HopSize = int(f_HopSize_sec*sr_hz)
    i_FFTSize = int(pow(2.0, mt.nextpow2(i_WinSize)))

    f_Win_v = scipy.signal.hamming(i_WinSize)
    f_SampRateX = float(sr_hz) / float(i_HopSize)
    f_SampRateY = i_FFTSize / sr_hz
    f_BinSize = sr_hz / i_FFTSize
    iHWinSize = int(np.fix((i_WinSize-1)/2))

    # === get input sig. (make analytic)
    f_Sig_v = audio_v

    #if np.isreal(f_Sig_v[0]):
    #    f_Sig_v = scipy.signal.hilbert(f_Sig_v);

    # === pre/post-pad signal
    f_Sig_v = np.concatenate((np.zeros(iHWinSize,float),  f_Sig_v, np.zeros(iHWinSize, float)))

    # === support vectors
    i_Len = len(f_Sig_v)
    i_Ind = np.arange(iHWinSize, i_Len-iHWinSize, i_HopSize)    #ind
    i_SizeX = len(i_Ind)
    i_SizeY = i_FFTSize
    f_SupX_v = np.arange(0, i_SizeX, 1.0) / f_SampRateX
    f_SupY_v = np.arange(0, i_SizeY, 1.0) / i_SizeY / 2.0

    """ calculate power spectrum """
    f_DistrPts_m    = np.zeros( (i_SizeY, i_SizeX), complex);

    for i in range(0, i_SizeX):
        #print "i_WinSize", i_WinSize, "iHWinSize", iHWinSize, "[a,b]", (i_Ind[i] - iHWinSize), (i_Ind[i]+iHWinSize), "\n"
        f_DistrPts_m[0:i_WinSize, i] = f_Sig_v[(i_Ind[i] - iHWinSize):(i_Ind[i] + iHWinSize + 1)] * f_Win_v

    # === fft (cols of dist.)
    """ Power distribution (pow) """
    X = scipy.fft(f_DistrPts_m, i_FFTSize, axis=0)
    S_pow = 1.0 / i_FFTSize * pow(abs(X), 2.0)
    S_pow = S_pow / sum(pow(f_Win_v, 2.))
    S_pow[1:, :] = S_pow[1:, :] / 2.

    S_mag = np.sqrt(1.0 / i_FFTSize) * abs(X)
    S_mag = S_mag / sum(abs(f_Win_v))
    S_mag[1:, :] = S_mag[1:, :] / 2.

    return S_mag, S_pow, i_SizeX, i_SizeY, f_SupX_v, f_SupY_v





def F_representationHarmonic(f_Sig_v, Fs):
    """
        Compute Harmonic representation (seems Ok)
        name: unknown
        @param
        @return
    """

    #print "F_representationHarmonic"

    f0_hz_v = []
    PartTrax_s = []
    f_DistrPts_m = []
    #f_SupX_v = []
    #f_SupY_v = []

    p_v, t_v, s_v = swp.swipep(f_Sig_v, float(Fs), np.array([50, 500]), 0.01, 1./48., 0.1, 0.2, -np.inf)
    ## for unitary test
    #m     = scipy.io.loadmat('test_f0.mat')
    #p_v    = np.array(m['p_v'])[:,0]
    #t_v    = np.array(m['t_v'])[:,0]
    #s_v    = np.array(m['s_v'])[:,0]

    # remove nan values
    i_n = (~np.isnan(p_v)).nonzero()[0]
    p_v = p_v[i_n]
    t_v = t_v[i_n]
    s_v = s_v[i_n]

    if max(s_v) > config_s.threshold_harmo:
        f0_bp = np.zeros((len(t_v), 2), float);
        f0_bp[:, 0] = t_v
        f0_bp[:, 1] = p_v
    else:
        f0_bp = []
        return f0_hz_v, f_DistrPts_m, PartTrax_s #, f_SupX_v, f_SupY_v;

    # ==========================================================
    # === Compute sinusoidal harmonic parameters
    L_sec = 0.1                                 # === analysis widow length
    STEP_sec = L_sec/4.                         # === hop size
    L_n = int(np.round(L_sec*Fs))
    STEP_n = int(np.round(STEP_sec*Fs))
    N = int(4*pow(2., mt.nextpow2(L_n)))        # === large zero-padding to get better frequency resolution
    fenetre_v = np.ones(L_n, float)             #scipy.signal.boxcar(L_n);
    fenetre_v = 2 * fenetre_v / sum(fenetre_v)

    B_m, F_v, T_v = mt.my_specgram(f_Sig_v, int(N), int(Fs), fenetre_v, int(L_n - STEP_n))   #mt.specgram
    B_m = abs(B_m)

    T_v = T_v + L_sec/2.
    nbFrame = np.shape(B_m)[1]
    f_DistrPts_m = pow(abs(B_m), 2.)
    #f_SupX_v = T_v;
    #f_SupY_v = F_v;
    lag_f0_hz_v = np.arange(-5, 5+0.1, 0.1)
    nb_delta = len(lag_f0_hz_v)
    inharmo_coef_v = np.arange(0, 0.001+0.00005, 0.00005)
    nbInharmo = len(inharmo_coef_v)
    totalenergy_3m = np.zeros((nbFrame, len(lag_f0_hz_v), len(inharmo_coef_v)), float)
    stock_pos_4m = np.zeros((nbFrame, len(lag_f0_hz_v), len(inharmo_coef_v), config_s.nb_harmo), int)

    f0_hz_v = F_evalbp(f0_bp, T_v)
    # === candidate_f0_hz_m (nbFrame, nb_delta)
    candidate_f0_hz_m = np.repeat(np.array([f0_hz_v, ]).T, nb_delta, axis=1) + np.repeat([lag_f0_hz_v, ], nbFrame, axis=0)
    stock_f0_m = candidate_f0_hz_m

    h = np.arange(1, config_s.nb_harmo+1., 1.)
    for numInharmo in range(0, nbInharmo):
        inharmo_coef = inharmo_coef_v[numInharmo]
        nnum_harmo_v = h * np.sqrt(1 + inharmo_coef * pow(h, 2.))

        for num_delta in range(0, nb_delta):
            # === candidate_f0_hz_v (nbFrame, 1)
            candidate_f0_hz_v = candidate_f0_hz_m[:, num_delta]
            # === candidate_f0_hz_m (nbFrame, nb_harmo): (nbFrame,1)*(1,nb_harmo)
            C1 = np.array([candidate_f0_hz_v, ]).T
            C2 = np.array([nnum_harmo_v, ])
            candidate_harmo_hz_m = np.dot(C1, C2)
            # === candidate_f0_hz_m (nbFrame, nb_harmo)
            candidate_harmo_pos_m = np.array( np.round(candidate_harmo_hz_m/Fs*N)+1, int)
            stock_pos_4m[:, num_delta, numInharmo, :] = candidate_harmo_pos_m
            for numFrame in range(0, nbFrame):
                totalenergy_3m[numFrame, num_delta, numInharmo] = np.sum(B_m[candidate_harmo_pos_m[numFrame, :], numFrame])

    # === choix du coefficient d'inharmonicite
    score_v = np.zeros(nbInharmo, float)
    for numInharmo in range(0, nbInharmo):
        score_v[numInharmo] = np.sum(np.max(np.squeeze(totalenergy_3m[:, :, numInharmo]), axis=0))

    ## pbm - test
    #print "score_v", score_v , "\n"
    #mt.my_plot(score_v)

    max_value, max_pos = mt.my_max(score_v)
    calcul = (score_v[max_pos]-score_v[0]) / (EPS + score_v[0])
    if calcul > 0.01:
        numInharmo = max_pos
    else:
        numInharmo = 1
    totalenergy_2m = np.squeeze(totalenergy_3m[:, :, numInharmo])

    PartTrax_s = np.zeros((nbFrame, 2, 20), float)
    for numFrame in range(0, nbFrame):
        max_value, num_delta = mt.my_max(totalenergy_2m[numFrame, :])
        f0_hz_v[numFrame] = stock_f0_m[numFrame, num_delta]
        cur_par = dPart
        f_Freq_v = np.squeeze(F_v[stock_pos_4m[numFrame, num_delta, numInharmo, :]])
        f_Ampl_v = B_m[stock_pos_4m[numFrame, num_delta, numInharmo, :], numFrame]
        PartTrax_s[numFrame, 0, :] = f_Freq_v
        PartTrax_s[numFrame, 1, :] = f_Ampl_v

    return f0_hz_v, f_DistrPts_m, PartTrax_s




def F_representationERB(f_Sig_v, Fs):
    """
        Compute ERB/Gammatone representation
        name: unknown
        @param
        @return
    """

    #print "F_representationERB"

    f_HopSize_sec = 256./44100.
    i_HopSize = f_HopSize_sec * Fs
    f_Exp = 0.25

    f_Sig_v_hat = outmidear(f_Sig_v, Fs)

    """ ERB """
    S_erb, f_erb, t_erb = ERBpower( f_Sig_v_hat, Fs, i_HopSize)
    S_erb = pow(S_erb, f_Exp); #cochleagram
    #loud_erb = np.sum(S_erb)
    #troids_erb = mt.centroid(S_erb)
    #troid_erb = np.sum(troids_erb * loud_erb) / np.sum(loud_erb)         # weighted average
    #centroid_erb = mt.my_interpolate(np.arange(0,nchans), f, troid_erb)        # to hz
    #centroids_erb = mt.my_interpolate(np.arange(0,nchans), f, troids_erb)        # to hz

    i_SizeX1 = len(t_erb)
    i_SizeY1 = len(f_erb)
    f_SupX_v1 = t_erb
    f_SupY_v1 = f_erb/Fs


    ## GAMMATONE
    S_gam, f_gam, t_gam = ERBpower2(f_Sig_v_hat, Fs, i_HopSize)
    S_gam = pow(S_gam, f_Exp); #cochleagram
    #loud_gam = np.sum(S_gam)
    #troids_gam = mt.centroid(S_gam)
    #troid_gam = np.sum(troids_gam * loud_gam) / np.sum(loud_gam)        # weighted average
    #centroid_gam = mt.my_interpolate(np.arange(0,nchans), f, troid_gam)       # to hz
    #centroids_gam = mt.my_interpolate(np.arange(0,nchans), f, troids_gam)       # to hz
    i_SizeX2 = len(t_gam)
    i_SizeY2 = len(f_gam)
    f_SupX_v2 = t_gam
    f_SupY_v2 = f_gam/Fs

    return S_erb, S_gam, i_SizeX1, i_SizeY1, f_SupX_v1, f_SupY_v1, i_SizeX2, i_SizeY2, f_SupX_v2, f_SupY_v2;



def ERBpower(a,Fs, hopsize, bwfactor=1.):
    """
        Compute ERB spectrum
        name: unknown
        @param
        @return
    """

    lo = 30.
    hi = 16000.
    hi = min(hi, (Fs/2.-mt.ERB(Fs/2.)/2.))
    nchans = np.round(2.*(mt.ERBfromhz(hi)-mt.ERBfromhz(lo)))
    cfarray = mt.ERBspace(lo, hi, nchans)
    nchans = len(cfarray)

    bw0 = 24.7;           # Hz - base frequency of ERB formula (= bandwidth of "0Hz" channel)
    b0 = bw0/0.982    # gammatone b parameter (Hartmann, 1997)
    ERD = 0.495 / b0     # based on numerical calculation of ERD
    wsize = int(pow(2., mt.nextpow2(ERD*Fs*2))[0])
    window = mt.gtwindow(wsize, wsize/(ERD*Fs))

    # pad signal with zeros to align analysis point with window power centroid
    m = len(a)
    offset = int(np.round(mt.centroid(pow(window, 2.)))[0])

    a = np.concatenate((np.zeros(offset), a, np.zeros(wsize-offset)))

    # matrix of windowed slices of signal
    fr, startsamples = mt.frames(a, wsize, hopsize)
    nframes = np.shape(fr)[1]
    fr = fr * np.repeat(np.array([window,]).T, nframes, axis=1)
    wh = int(np.round(wsize/2.))

    # power spectrum
    pwrspect = pow(abs(scipy.fft(fr, int(wsize), axis=0)), 2.)
    pwrspect = pwrspect[0:wh, :]

    # array of kernel bandwidth coeffs:
    b = mt.ERB(cfarray) / 0.982
    b = b * bwfactor
    bb = np.sqrt(pow(b, 2.) - pow(b0, 2.))

    # matrix of kernels (array of gammatone power tfs sampled at fft spectrum frequencies).
    iif = np.array([np.arange(1, wh+1),])

    f = np.repeat(iif * Fs / wsize, nchans,  axis=0).T
    cf = np.repeat(np.array([cfarray,]), wh, axis=0)
    bb = np.repeat(np.array([bb,]), wh, axis=0)

    wfunct = pow( abs(1./pow(1j * (f - cf) + bb, 4.)), 2.)
    adjustweight = mt.ERB(cfarray) / sum(wfunct)
    wfunct = wfunct * np.repeat(np.array([adjustweight,]), wh, axis=0)
    wfunct = wfunct / np.max(np.max(wfunct))

    # multiply fft power spectrum matrix by weighting function matrix:
    c = np.dot(wfunct.T, pwrspect)
    f = cfarray
    t = startsamples/Fs
    return c, f, t



def ERBpower2(a, Fs, hopsize, bwfactor=1.):
    """
        Compute Gammatone spectrum
        name: unknown
        @param
        @return
    """

    lo        = 30.;                            # Hz - lower cf
    hi        = 16000.;                         # Hz - upper cf
    hi        = min(hi, (Fs/ 2. - np.squeeze(mt.ERB( Fs / 2.)) / 2. )); # limit to 1/2 erb below Nyquist

    nchans    = np.round( 2. * (mt.ERBfromhz(hi)-mt.ERBfromhz(lo)) );
    cfarray = mt.ERBspace(lo,hi,nchans);
    nchans = len(cfarray);

    # apply gammatone filterbank
    b = mt.gtfbank(a, Fs, cfarray, bwfactor);

    # instantaneous power
    b = mt.fbankpwrsmooth(b, Fs, cfarray);

    # smooth with a hopsize window, downsample
    b            = mt.rsmooth(b.T, hopsize, 1, 1);

    b            = np.maximum(b.T, 0);
    m,n            = np.shape(b);

    nframes        = np.floor( float(n) / hopsize );
    startsamples= np.squeeze(np.array(np.round(np.arange(0, nframes, 1) * int(hopsize)), int));

    c            = b[:,startsamples];
    f             = cfarray;
    t             = startsamples / Fs;

    return c, f, t



def F_evalbp(bp, x_v):
    """
    """

    y_v = np.zeros(len(x_v), float)
    pos1 = (x_v < bp[0,0]).nonzero()[0]
    if len(pos1) > 0:
        y_v[pos1] = bp[0,1]
    pos2 = (x_v > bp[np.shape(bp)[0]-1,0]).nonzero()[0]
    if len(pos2)>0:
        y_v[pos2] = bp[np.shape(bp)[0]-1, 1]
    pos  =  ((x_v >= bp[0,0]) & (x_v <= bp[np.shape(x_v)[0]-1, 1])).nonzero()[0]

    if len(x_v[pos]) > 1:
        y_v[pos] = mt.my_interpolate( bp[:,0], bp[:,1], x_v[pos], 'linear')
    else:
        for n in range(0, len(x_v)):
            x = x_v[n]
            min_value, min_pos = mt.my_min( abs( bp[:, 0] - x))
            L = np.shape(bp)[0]
            t1    = (bp[min_pos, 0] == x) or (L == 1)
            t2    = (bp[min_pos, 0] < x) and (min_pos == L)
            t3    = (bp[min_pos, 0] > x) and (min_pos == 0)
            if t1 or t2 or t3:
                y_v[n] = bp[min_pos,1]
            else:
                if bp[min_pos, 0] < x:
                    y_v[n] = (bp[min_pos+1, 0] - bp[min_pos, 0]) / (bp[min_pos+1,0] - bp[min_pos,0]) * (x - bp[min_pos, 0]) + bp[min_pos, 0]
                else:
                    if bp[min_pos, 0] > x:
                        y_v[n] = (bp[min_pos, 1] - bp[min_pos-1,1]) / (bp[min_pos, 0] - bp[min_pos-1, 0]) * (x - bp[min_pos-1,0]) + bp[min_pos-1,1]

    return y_v















####################################################################################
####                           Sub-functions                                    ####
####################################################################################
#  y = outmidear(x, Fs) - Tested OK
#  name: unknown
#  @param
#  @return
#
def outmidear(x, Fs):
    maf, f    = isomaf([], 'killion');                        # minimum audible field sampled at f
    g,tg     = isomaf([1000])-maf;                            # gain re: 1kHz
    g        = pow(10., g/20.);                                # dB to lin
    f        = np.concatenate( ([0], f, [20000]) );        # add 0 and 20 kHz points
    g = np.concatenate( ([EPS], g, [ g[len(g)-1]] ));    # give them zero amplitude

    if (Fs/2.) > 20000:
        f    = np.concatenate( ( f, np.array( [ np.squeeze(Fs)/2.])  ));
        g    = np.concatenate( ( g, np.array([g[len(g)-1]]) ));

    # Model low frequency part with 2 cascaded second-order highpass sections:
    fc     = 680.;                                                 # Hz - corner frequency
    q    = 0.65;                                                    # quality factor
    pwr    = 2;                                                    # times to apply same filter
    a    = sof( fc / Fs, q);                                        # second order low-pass
    b    = np.concatenate( ([sum(a)-1], [-a[1]], [-a[2]]) );    # convert to high-pass

    for k in range(0, pwr):
        x    = scipy.signal.lfilter(b,a,x);

    # Transfer function of filter applied:
    ff, gg    = scipy.signal.freqz(b,a);
    gg        = pow(abs(gg), float(pwr));
    ff        = ff * Fs / (2.* np.pi);

    # Transfer function that remains to apply:
    g                                 = mt.my_interpolate(f, g, ff, 'linear');
    gain                            = g / (gg+EPS);
    gain[ (ff<f[1]).nonzero()[0] ]    = 1.;
    N                                 = 51.;    # order
    lg                                 = np.linspace( 0., 1., len(gain) );
    #ipdb.set_trace()
    b                                 = scipy.signal.firwin2(int(N), lg, gain);
    return scipy.signal.lfilter(b,1.,x);




def sof(f,q):
    # a=sof(f,q) - second-order lowpass filter
    #
    # f: normalized resonant frequency (or column vector of frequencies)
    # q: quality factor (or column vector of quality factors)
    #
    # a: filter coeffs
    #
    # based on Malcolm Slaney's auditory toolbox
    rho     = np.exp(-np.pi * f / q);
    theta     = 2.*np.pi * f *np.sqrt(1. - 1. / (4 * pow(q, 2.)));

    return np.concatenate( (np.ones(len(rho),float), -2. * rho * np.cos(theta), pow(rho, 2.)));




def isomaf(f=[], dataset='killion'):
    if dataset == 'moore':
        freqs     = np.array([0,20.,25.,31.5,40.,50.,63.,80.,100.,125.,160.,200.,250.,315.,400.,500.,630.,800.,1000.,1250.,1600.,2000.,2500.,3150.,4000.,5000.,6300.,8000.,10000.,12500.,15000.,20000.]);
        datamaf = np.array([75.8,70.1,60.8,52.1,44.2,37.5,31.3,25.6,20.9,16.5,12.6,9.6,7.0,4.7,3.0,1.8,0.8,0.2,0.0,-0.5,-1.6,-3.2,-5.4,-7.8,-8.1,-5.3,2.4,11.1,12.2,7.4,17.8,17.8]);
        freqs    = freqs[1:(len(freqs)-1)];
        datamaf    = datamaf[1:(len(datamaf)-1)];
    else:
        freqs     = np.array([100.,150.,200.,300.,400.,500.,700.,1000.,1500.,2000.,2500.,3000.,3500.,4000.,4500.,5000.,6000.,7000.,8000.,9000.,10000.]);
        datamaf = np.array([33.,24.,18.5,12.,8.,6.,4.7,4.2,3.,1.,-1.2,-2.9,-3.9,-3.9,-3.,-1.,4.6,10.9,15.3,17.,16.4]);

    if len(f) < 1:
        f        = freqs;
        mafs    = datamaf;
    else:
        mafs     = mt.my_interpolate(freqs, datamaf, f, 'linear'); #'cubic'
    # for out of range queries use closest sample
    I1             = (f<min(freqs)).nonzero()[0];
    mafs[I1]    = datamaf[0];
    I2            = (f>max(freqs)).nonzero()[0];
    mafs[I2]    = datamaf[len(datamaf)-1];
    return mafs,f




def F_computeLogAttack(f_Env_v, Fs, f_ThreshNoise):
    Fs             = float(Fs);
    my_eps         = pow(10.0,-3.0);                  #increase if errors occur
    f_ThreshDecr    = 0.4;
    percent_step    = 0.1;
    param_m1        = int(round(0.3/percent_step));     # === BORNES pour calcul mean
    param_m2        = int(round(0.6/percent_step));
    param_s1att    = int(round(0.1/percent_step));     # === BORNES pour correction satt (start attack)
    param_s2att    = int(round(0.3/percent_step));
    param_e1att    = int(round(0.5/percent_step));     # === BORNES pour correction eatt (end attack)
    param_e2att    = int(round(0.9/percent_step));
    param_mult    = 3.0;                               # === facteur multiplicatif de l'effort

    # === calcul de la pos pour chaque seuil
    f_EnvMax        = max(f_Env_v);                    #i_EnvMaxInd
    f_Env_v        = f_Env_v / (f_EnvMax+EPS);         # normalize by maximum value

    percent_value_v     = mt.my_linspace(percent_step, 1.0, percent_step); # np.arange(percent_step, 1.0, percent_step)
    nb_val             = int(len(percent_value_v));
    #np.linspace(percent_step, 1, nb_val);
    #nb_val = int(np.round(1.0/ percent_step));
    percent_value_v = np.linspace(percent_step, 1, nb_val);
    percent_posn_v    = np.zeros(nb_val, int);

    for p in range(0, nb_val):
        pos_v = (f_Env_v >= percent_value_v[p]-my_eps).nonzero()[0];
        if len(pos_v) > 0:
            percent_posn_v[p]    = pos_v[0];
        else:
            x, percent_posn_v[p] = mt.my_max(f_Env_v);

    # === NOTATION
    # satt: start attack
    # eatt: end attack
    #==== detection du start (satt_posn) et du stop (eatt_posn) de l'attaque
    pos_v             = (f_Env_v > f_ThreshNoise).nonzero()[0];
    dpercent_posn_v    = np.diff(percent_posn_v);
    M                = np.mean(dpercent_posn_v[(param_m1-1):param_m2]);

    # === 1) START ATTACK
    pos2_v    = ( dpercent_posn_v[ (param_s1att-1):param_s2att ] > param_mult*M ).nonzero()[0];
    if len(pos2_v) > 0:
        result    = pos2_v[len(pos2_v)-1]+param_s1att+1;
    else:
        result    = param_s1att;
    result = result -1;

    satt_posn    = percent_posn_v[result];

    # === raffinement: on cherche le minimum local
    n        = percent_posn_v[result];
    delta    = int(np.round(0.25*(percent_posn_v[result+1]-n)));

    if n-delta >= 0:
        min_value, min_pos    = mt.my_min(f_Env_v[ int(n-delta):int(n+delta) ]);
        satt_posn            = min_pos + n-delta;

    # === 2) END ATTACK
    pos2_v    = (dpercent_posn_v[(param_e1att-1):param_e2att] > param_mult*M).nonzero()[0];
    if len(pos2_v) >0:
        result    = pos2_v[0]+param_e1att-1;
    else:
        result    = param_e2att-1;

    # === raffinement: on cherche le maximum local
    delta    = int(np.round(0.25*( percent_posn_v[result] - percent_posn_v[result-1] )));
    n        = percent_posn_v[result];
    if n+delta < len(f_Env_v):
        #print "n", n, " delta", delta, "len(f_Env_v)", len(f_Env_v),"\n"
        max_value, max_pos    = mt.my_max(f_Env_v[int(n-delta):int(n+delta)]);
        eatt_posn            = max_pos + n-delta;#-1;

    # === D: Log-Attack-Time
    if satt_posn == eatt_posn:
        satt_posn = satt_posn - 1;

    risetime_n    = (eatt_posn - satt_posn);
    f_LAT          = np.log10(risetime_n/Fs);

    # === D: croissance temporelle
    satt_value        = f_Env_v[satt_posn];
    eatt_value        = f_Env_v[eatt_posn];
    seuil_value_v    = np.arange(satt_value, eatt_value, 0.1);

    seuil_possec_v    = np.zeros(len(seuil_value_v), float);
    for p in range(0, len(seuil_value_v)):
        pos3_v                = ( f_Env_v[ satt_posn:(eatt_posn+1) ] >= seuil_value_v[p] ).nonzero()[0];
        seuil_possec_v[p]    = pos3_v[0]/Fs;


    pente_v            = np.diff( seuil_value_v) / np.diff(seuil_possec_v);
    mseuil_value_v    = 0.5*(seuil_value_v[0:(len(seuil_value_v)-1)]+seuil_value_v[1:]);
    weight_v            = np.exp( -pow(mseuil_value_v-0.5,2.0) / 0.25);
    f_Incr            = np.sum( pente_v * weight_v) / (EPS+np.sum(weight_v));
    tempsincr            = np.arange( satt_posn, eatt_posn+1);
    tempsincr_sec_v    = tempsincr / Fs;
    const            = np.mean( f_Env_v[ np.round(tempsincr)] - f_Incr*tempsincr_sec_v);
    mon_poly_incr        = np.concatenate(([f_Incr],[const]));
    mon_poly_incr2    = np.polyfit(tempsincr_sec_v, f_Env_v[tempsincr], 1);
    incr2            = mon_poly_incr2[0];

    # === D: decroissance temporelle
    fEnvMax, iEnvMaxInd = mt.my_max(f_Env_v);
    iEnvMaxInd        = round(0.5*(iEnvMaxInd+eatt_posn));
    pos_v            = (f_Env_v > f_ThreshDecr).nonzero()[0];
    stop_posn            = pos_v[len(pos_v)-1];

    if iEnvMaxInd == stop_posn:
        if stop_posn < len(f_Env_v):
            stop_posn = stop_posn+1;
        else:
            if iEnvMaxInd>1:
                iEnvMaxInd = iEnvMaxInd-1;

    tempsdecr            = np.array(range(int(iEnvMaxInd), int(stop_posn+1)));
    tempsdecr_sec_v    = tempsdecr / Fs;
    mon_poly_decr     = np.polyfit( tempsdecr_sec_v, np.log( f_Env_v[tempsdecr]+mt.EPS), 1);
    f_Decr               = mon_poly_decr[0];

    # === D: enveloppe ADSR = [A(1) | A(2)=D(1) | D(2)=S(1) | S(2)=D(1) | D(2)]
    f_ADSR_v    = np.array( [satt_posn, iEnvMaxInd, 0.0, 0.0, stop_posn] ) / Fs;
    return f_LAT, f_Incr, f_Decr, f_ADSR_v



def F_computeTempCentroid(f_Env_v, f_Thresh):

    f_MaxEnv, i_MaxInd    = mt.my_max(f_Env_v);
    f_Env_v    = f_Env_v / f_MaxEnv;       # normalize
    i_Pos_v    = (f_Env_v > f_Thresh).nonzero()[0];
    i_StartFrm = i_Pos_v[0];
    if i_StartFrm == i_MaxInd:
        i_StartFrm = i_StartFrm - 1;
    i_StopFrm         = i_Pos_v[len(i_Pos_v)-1];
    f_Env2_v        = f_Env_v[range(i_StartFrm, i_StopFrm+1)];
    f_SupVec_v    = np.array(range(1, len(f_Env2_v)+1))-1;
    f_Mean        = np.sum( f_SupVec_v * f_Env2_v) / np.sum(f_Env2_v);  # centroid
    f_TempCent    = (i_StartFrm + f_Mean);            # temporal centroid (in samples)
    return f_TempCent




def F_computeEffectiveDuration(f_Env_v, f_Thresh):

    f_MaxEnv, i_MaxInd    = mt.my_max(f_Env_v);            #=== max value and index
    f_Env_v        = f_Env_v / f_MaxEnv;        # === normalize
    i_Pos_v        = (f_Env_v > f_Thresh).nonzero()[0];

    i_StartFrm = i_Pos_v[0];
    if i_StartFrm == i_MaxInd:
        i_StartFrm = i_StartFrm - 1;
    i_StopFrm        = i_Pos_v[len(i_Pos_v)-1];
    f_EffDur        = (i_StopFrm - i_StartFrm + 1);
    return f_EffDur;




def F_computeModulation(f_Env_v, f_ADSR_v, Fs):

    # do.method        = 'fft'; % === 'fft' 'hilbert'
    sustain_Thresh = 0.02;  #in sec
    envelopfull_v    = f_Env_v;
    tempsfull_sec_v = np.arange(0, len(f_Env_v), 1.) / Fs;

    sr_hz    = 1.0/np.mean( np.diff(tempsfull_sec_v));
    ss_sec    = f_ADSR_v[1];                 # === start sustain
    es_sec    = f_ADSR_v[4];                 # === end   sustain

    flag_is_sustained = 0;
    if (es_sec - ss_sec) > sustain_Thresh:  # === if there is a sustained part
        a     = (ss_sec <= tempsfull_sec_v).nonzero()[0];
        b     = (tempsfull_sec_v <= es_sec).nonzero()[0];
        pos_v= ( (ss_sec <= tempsfull_sec_v) & (tempsfull_sec_v <= es_sec)).nonzero()[0];
        if len(pos_v) > 0:
            flag_is_sustained = 1;

    if flag_is_sustained == 1:
        envelop_v    = envelopfull_v[pos_v];
        temps_sec_v    = tempsfull_sec_v[pos_v];
        M            = np.mean(envelop_v);

        # === TAKING THE ENVELOP
        mon_poly    = np.polyfit(temps_sec_v, np.log(envelop_v+EPS), 1);
        hatenvelop_v= np.exp(np.polyval(mon_poly, temps_sec_v));
        signal_v    = envelop_v - hatenvelop_v;

        L_n            = len(signal_v);
        N           = int(round(max(np.array([sr_hz, pow(2.0, mt.nextpow2(L_n))]))));
        fenetre_v    = scipy.hamming(L_n);
        norma        = np.sum(fenetre_v) ;
        fft_v        = scipy.fft(signal_v * fenetre_v*2/norma, N);
        ampl_v         = np.abs(fft_v);
        phas_v        = np.angle(fft_v);
        freq_v         = mt.Index_to_freq( np.arange(0,N,1, float), sr_hz, N);

        param_fmin    = 1.0;
        param_fmax     = 10.0;
        pos_v          = ( (freq_v < param_fmax) & (freq_v > param_fmin) ).nonzero()[0];

        pos_max_v    = F_comparepics2(ampl_v[pos_v], 2);

        if len(pos_max_v) > 0:
            max_value, max_pos    = mt.my_max( ampl_v[ pos_v[ pos_max_v]]);
            max_pos                = pos_v[pos_max_v[max_pos]];
        else:
            max_value, max_pos    = mt.my_max(ampl_v[pos_v]);
            max_pos             = pos_v[max_pos];

        MOD_am        = max_value/(M+EPS);
        MOD_fr        = freq_v[max_pos];
        MOD_ph        = phas_v[max_pos];

        #if len(max_pos) == 0:#(len(MOD_am)==0) | (len(MOD_fr)==0):
        #    MOD_am = 0;
        #    MOD_fr = 0;

    ## disabled (Hilbert Method)
        #case 'hilbert', % ==========================
            #sa_v        = np.hilbert(signal_v(:));
            #sa_ampl_v    = abs(signal_v);
            #sa_phase_v    = unwrap(angle(hilbert(signal_v)));
            #sa_freqinst_v= 1/(2*pi)*sa_phase_v./(temps_n_v/sr_hz);

            #MOD_am        = median(sa_ampl_v);
            #MOD_fr        = median(sa_freqinst_v);

    else:   # === if there is NO  sustained part
        MOD_fr = 0;
        MOD_am = 0;

    return MOD_fr, MOD_am




def F_comparepics2(input_v, lag_n=2, do_affiche=0, lag2_n=0, seuil=0):
    if lag2_n == 0:
        lag2_n = 2*lag_n;
    L_n         = len(input_v);
    pos_cand_v     = (np.diff( np.sign( np.diff(input_v))) < 0).nonzero()[0];

    pos_cand_v = pos_cand_v+1;
    pos_max_v     = np.array([],int);

    for p in range(0, len(pos_cand_v)):
        pos = pos_cand_v[p];
        i1 = (pos-lag_n);
        i2 = (pos+lag_n+1);
        i3 = (pos-lag2_n);
        i4 = (pos+lag2_n+1);

        if (i1 >= 0) and (i2 <= L_n):
            tmp                    = input_v[i1:i2];
            maximum, position    = mt.my_max(tmp);
            position            = position + i1;

            if (i3>=0) and (i4<=L_n):
                tmp2            = input_v[i3:i4];
                if (position == pos) and (input_v[position] > seuil*np.mean(tmp2)):
                    pos_max_v = np.concatenate( (pos_max_v, np.array([pos])) );

    if lag_n < 2:
        if input_v[0] > input_v[1]:
            pos_max_v = np.arange(0,pos_max_v);
        if input_v[len(input_v)-1] > input_v[len(input_v)-1]:
            pos_max_v = np.concatenate( (pos_max_v, np.array([L_n])));

    return pos_max_v;



# ==========================
def main(argv):
    filename = '/Users/peeters/_work/_sound/_collection/adobe_soundFX/Cartoon/Cartoon Balloon Air Release 01.wav'
    sr_hz, audio_m = wavfile.read(filename)
    audio_v = np.mean(audio_m, axis=1)
    descHub_d = F_computeAllDescriptor(audio_v, sr_hz)
    descHub_d = F_temporalModeling(descHub_d)
    ipdb.set_trace()


if __name__ == '__main__':
    main(sys.argv[1:])
