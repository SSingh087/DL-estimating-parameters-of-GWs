import numpy as np
from pycbc.waveform import get_td_waveform
from tqdm import tqdm
import csv
import h5py
import pandas as pd
from scipy import signal
import scipy.io.wavfile as s

next_val=0

#------------------------------------------------------------------
#                    TRAINING DATASETS PREPRATION               ---
#------------------------------------------------------------------

class dataprep_train:
  
  def __init__():
    None
    
#------------------------------------------------------------------
#                    SIMULATED SIGNALS+TRANSIENT NOISE
#------------------------------------------------------------------
  def simulated_signals(noise):  
    apx = ['TaylorT1','TaylorT2','EOBNRv2','SEOBNRv1','SEOBNRv2']
    with open('gdrive/My Drive/GW data/labels.csv', 'a', newline='') as file:
      for a in tqdm(range(len(apx))):
          ...
          ...
          ...
          hp, hc = get_td_waveform(.., .., .., ..)
              
          signal_gw[next_val:next_val+noise.shape[0],:] = np.copy(noise)
          pos=np.random.randint(0,noise.shape[1]-len(hp))
          signal_gw[next_val:next_val+noise.shape[0],pos:pos+len(hp)]+=hp
          writer = csv.writer(file)
            #--------update next_val ----------#
          next_val+=(noise.shape[0])


#------------------------------------------------------------------
#                    ECHOES +TRANSIENT NOISE
#------------------------------------------------------------------
  def echoes(noise):
  
    signal_gw[next_val:next_val+noise.shape[0],:] = np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for loop in tqdm(range(10)):
        t=np.linspace(0,.3,np.random.randint(noise.shape[1]))
        y1, y2=np.zeros(len(t)),np.zeros(len(t))
        i=0
        for j in range(10):
          for f in [250,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295
              y1[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            ..
            ..
#------------ Inject signal----------------
            ..
            ..
#-------------update next_val------------------
            ..
            ..
#--------------write to csv-------------------


        r=.3
        for j in range(10):
          for f in [250,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
              y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            ..
            ..         
            ..
#------------ Inject signal----------------
            ..
            ..
#-------------update next_val------------------
            ..
            ..
#--------------write to csv-------------------


#------------------------------------------------------------------
#                    GLITCHES+TRANSIENT NOISE
#------------------------------------------------------------------
  def glitches(noise):
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for i in tqdm(...13 .. samples ...):
        ..
        ..
#-------filter signal using scipy -------------------
          ..
          ..
#------------ Inject signal----------------
          ..
          ..
#-------------update next_val------------------
          ..
          ..
#--------------write to csv-------------------

#------------------------------------------------------------------------------
#                             CCSNe+Transient Noise               -------------
#------------------------------------------------------------------------------
  def ccsne(noise):
    signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      val=['signal_A1B1G1_R.dat','signal_A1B2G1_R.dat','signal_A1B3G1_R.dat','signal_A1B3G2_R.dat','signal_A1B3G3_R.dat','signal_A1B3G5_R.dat','signal_A2B4G1_R.dat','signal_A3B1G1_R.dat','signal_A3B2G1_R.dat','signal_A3B2G2_R.dat','signal_A3B2G4_soft_R.dat','signal_A3B2G4_R.dat','signal_A3B3G1_R.dat','signal_A3B3G2_R.dat','signal_A3B3G3_R.dat','signal_A3B3G5_R.dat','signal_A3B4G2_R.dat','signal_A3B5G4_R.dat','signal_A4B1G1_R.dat','signal_A4B1G2_R.dat','signal_A4B2G2_R.dat','signal_A4B2G3_R.dat','signal_A4B4G4_R.dat','signal_A4B4G5_R.dat','signal_A4B5G4_R.dat','signal_A4B5G5_R.dat']
      for aak in tqdm(val):
        loc='gdrive/My Drive/GW data/CCSNe/'+aak
        x, y = np.loadtxt(loc,unpack=True, usecols=[0,1])
        for r in [10,30]:
          for theta in [30,60]:
            y = 1/8*np.sqrt(15/np.pi)*y/r*(np.sin(theta))**2
          ..
          ..
#------------ Inject signal----------------
          ..
          ..
#-------------update next_val------------------
          ..
          ..
#--------------write to csv-------------------

#------------------------------------------------------------------------------
#                             MIXED SIGNALS  BBH/BNS/GW-Echoes     ------------
#------------------------------------------------------------------------------
  def mixed_signals_BHBNSB(noise):
    apx = ['TaylorT1','TaylorT2','EOBNRv2','SEOBNRv1','SEOBNRv2']
    with open('gdrive/My Drive/GW data/labels.csv', 'a', newline='') as file:
      for a in tqdm(range(len(apx))):
          ...
          ...
          ...
          hp, hc = get_td_waveform(.., .., .., ..)
          ..
          ..
#-------------for each signal mix GW-echoes----------------------------
        ##-------------------CIE--------------------------
          ...
          ...
            signal_gw[next_val:next_val+noise.shape[0],:] = np.copy(noise)
            with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
              for loop in tqdm(range(10)):
                t=np.linspace(0,.3,np.random.randint(noise.shape[1]))
                y1, y2=np.zeros(len(t)),np.zeros(len(t))
                i=0
                for j in range(10):
                  for f in [250,280]:
                    for i in range(len(t)):
                      aa=t[i]-0.0295-j*0.0295
                      y1[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
                  ..
                  ..
      #-------------For each signal add glitch ----------------------------
                  ..
                  ..
                  
                  signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
                  with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
                    for i in tqdm(...13 .. samples ...):
                      ..
                      ..
              #-------filter signal using scipy -------------------
                        ..
                        ..
              #------------ Inject signal----------------
                        ..
                        ..
              #-------------update next_val------------------
                        ..
                        ..
              #--------------write to csv-------------------
          
          ##--------------------------REPEAT FOR UIE-----------------------------------


#------------------------------------------------------------------------------
#                             MIXED SIGNALS CCSNe                 -------------
#------------------------------------------------------------------------------
  def mixed_signals_CCSNe(noise):
    signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      val=['signal_A1B1G1_R.dat','signal_A1B2G1_R.dat','signal_A1B3G1_R.dat','signal_A1B3G2_R.dat','signal_A1B3G3_R.dat','signal_A1B3G5_R.dat','signal_A2B4G1_R.dat','signal_A3B1G1_R.dat','signal_A3B2G1_R.dat','signal_A3B2G2_R.dat','signal_A3B2G4_soft_R.dat','signal_A3B2G4_R.dat','signal_A3B3G1_R.dat','signal_A3B3G2_R.dat','signal_A3B3G3_R.dat','signal_A3B3G5_R.dat','signal_A3B4G2_R.dat','signal_A3B5G4_R.dat','signal_A4B1G1_R.dat','signal_A4B1G2_R.dat','signal_A4B2G2_R.dat','signal_A4B2G3_R.dat','signal_A4B4G4_R.dat','signal_A4B4G5_R.dat','signal_A4B5G4_R.dat','signal_A4B5G5_R.dat']
      for aak in tqdm(val):
        loc='gdrive/My Drive/GW data/CCSNe/'+aak
        x, y = np.loadtxt(loc,unpack=True, usecols=[0,1])
        for r in [10,30]:
          for theta in [30,60]:
            y = 1/8*np.sqrt(15/np.pi)*y/r*(np.sin(theta))**2

      #-------------For each signal add glitch ----------------------------
                  ..
                  ..
                  
                  signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
                  with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
                    for i in tqdm(...13 .. samples ...):
                      ..
                      ..
              #-------filter signal using scipy -------------------
                        ..
                        ..
              #------------ Inject signal----------------
                        ..
                        ..
              #-------------update next_val------------------
                        ..
                        ..
              #--------------write to csv-------------------
          

#----------------------------------------------------------------------------------
#                               PIPELINES                                       ---
#----------------------------------------------------------------------------------
def train_pipeline(noise):

  val=dataprep_train
  with open('gdrive/My Drive/GW data/labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([.., .., .., .., .., #------ADD 23 labels---------])
  print('\nSimulated GW .......')    
  val.simulated_signals(noise)
  print('\nSimulated GW training set 100%')
  print('data size :'+ str(next_val)+'\n')  
  print('\nEchoes...')
  val.echoes(noise)
  print('data size :'+ str(next_val)+'\n')
  print('\nEchoes 100%')
  print('\nCCSNE...')  
  val.ccsne(noise)
  print('data size :'+ str(next_val)+'\n')
  print('\nCCSNE 100%')
  print('\nGlitches...')    
  val.glitches(noise)
  print('data size :'+ str(next_val)+'\n')  
  print('\nGlitches 100%')  
  print('\nMixed training BHBNSB ...' )
  val.mixed_signals_BHBNSB(noise)
  print('\nMixed set BHBNSB 100%')
  print('data size :'+ str(next_val)+'\n')
  print('\nMixed training CCSNe...' )
  val.mixed_signals_CCSNe(noise)
  print('\nMixed set CCSNe 100%')
  print('data size :'+ str(next_val)+'\n')
  hf = h5py.File('gdrive/My Drive/GW data/data.h5', 'w')
  hf.create_dataset('data', data=signal_gw)
  hf.close()


#--------------------------------------------------------
#                           MAIN                    -----
#--------------------------------------------------------
if __name__ == '__main__':
  hf= h5py.File('gdrive/My Drive/GW data/noise_data.hdf5', 'r')
  group_key = list(hf.keys())
  strain=hf['strain']['Strain'].value
  ts = hf['strain']['Strain'].attrs['Xspacing']
  metaKeys = hf['meta'].keys()
  meta = hf['meta']
  gpsStart = meta['GPSstart'].value
  duration = meta['Duration'].value
  gpsEnd   = gpsStart + duration
  time = np.arange(gpsStart, gpsEnd, ts)

# --------------- define noise segment------------------- #

  print('\n\nPreparing data..... ')
  train_pipeline(noise)
  print('\nPreparing data........100%\n\n')

#------------------normalize data points ------------------ #
  ..
  ..
#------------------reshape array into nxnx3 dims ------------------ #
  ..
  ..
