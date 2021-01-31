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
          Inject signal
          ..
          ..
          update next_val
          ..
          ..
          write to csv

        r=.3
        for j in range(10):
          for f in [250,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
              y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            ..
            ..
            Inject signal
            ..
            ..
            update next_val
            ..
            ..
            write to csv


#------------------------------------------------------------------
#                    GLITCHES+TRANSIENT NOISE
#------------------------------------------------------------------
  def glitches(noise):
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for i in tqdm(...13 .. samples ...):
        ..
        ..
        filter signal using scipy 
        ..
        ..
        Inject signal
        ..
        ..
        update next_val
        ..
        ..
        write to csv

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
          Inject signal
          ..
          ..
          update next_val
          ..
          ..
          write to csv
#------------------------------------------------------------------------------
#                             MIXED SIGNALS                       -------------
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


#------------------------------------------------------------------------------
#                             MIXED SIGNALS                       -------------
#------------------------------------------------------------------------------
  def mixed_signals_CCSNe(noise):
    global next_val,signal_gw  
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      val=['signal_A1B1G1_R.dat','signal_A1B2G1_R.dat','signal_A1B3G1_R.dat','signal_A1B3G2_R.dat','signal_A1B3G3_R.dat','signal_A1B3G5_R.dat','signal_A2B4G1_R.dat','signal_A3B1G1_R.dat','signal_A3B2G1_R.dat','signal_A3B2G2_R.dat','signal_A3B2G4_soft_R.dat','signal_A3B2G4_R.dat','signal_A3B3G1_R.dat','signal_A3B3G2_R.dat','signal_A3B3G3_R.dat','signal_A3B3G5_R.dat','signal_A3B4G2_R.dat','signal_A3B5G4_R.dat','signal_A4B1G1_R.dat','signal_A4B1G2_R.dat','signal_A4B2G2_R.dat','signal_A4B2G3_R.dat','signal_A4B4G4_R.dat','signal_A4B4G5_R.dat','signal_A4B5G4_R.dat','signal_A4B5G5_R.dat']
      for aak in tqdm(val):
        loc='gdrive/My Drive/GW data/CCSNe/'+aak
        x, y = np.loadtxt(loc,unpack=True, usecols=[0,1])
        for r in [10,30]:
          for theta in [30,60]:
            y = 1/8*np.sqrt(15/np.pi)*y/r*(np.sin(theta))**2
            new_arr=np.zeros(noise.shape[1]-500)
            j=0
            for i in range(0,len(y),2):
              new_arr[j]=y[i]
              j+=1               
            wave=['helix2','whistle','wandering_line','violin_mode','Tomte','koi_fish','scratchy','scattered_light','repeating_blip','power_line','paired_dove','low_freq_burst','light_modulation']
            for wave_name in wave: 
              loc='gdrive/My Drive/GW data/Glitches/'+wave_name+'.wav'
              rate,data=s.read(loc)
              for c2 in [5]:
                b, a = signal.butter(7, .9, btype='lowpass', analog=False)
                low_passed = signal.filtfilt(b, a, data)
                z = 10e-28*signal.medfilt(low_passed,c2)
                signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise)
                pos=np.random.randint(0,noise.shape[1]-len(z)) 
                signal_gw[next_val:next_val+noise.shape[0],pos:pos+len(z)]+=z 
                pos=np.random.randint(0,noise.shape[1]-len(new_arr))
                signal_gw[next_val:next_val+noise.shape[0],pos:pos+len(new_arr)]+=new_arr   
                writer = csv.writer(file)
                for i in range(noise.shape[0]):
                  col=np.zeros(27)
                  if val=='signal_A2B4G1_R.dat'  or   val=='signal_A3B3G5_R.dat' or  val=='signal_A3B5G4_R.dat' :
                    col[19]=1
                  elif val=='signal_A3B3G1_R.dat'  or val=='signal_A3B4G2_R.dat':
                    col[20]=1
                  elif val=='signal_A3B3G2_R.dat' or  val=='signal_A3B3G3_R.dat' or  val=='signal_A4B2G2_R.dat' or  val=='signal_A4B2G3_R.dat' or  val=='signal_A4B4G4_R.dat' or  val=='signal_A4B4G5_R.dat' or  val=='signal_A4B5G4_R.dat' or  val=='signal_A4B5G5_R.dat':
                    col[20],col[21]=1,1
                  else:
                    col[21]=1
                  if r==10:
                    col[22]=1
                  else:
                    col[23]=1
                  col[18],col[23+theta//30],col[26]=1,1,1
                  writer.writerow(col)
                next_val+=noise.shape[0]



#----------------------------------------------------------------------------------
#                               PIPELINES                                       ---
#----------------------------------------------------------------------------------
def train_pipeline(noise):
  global next_val,signal_gw
  val=dataprep_train
  with open('gdrive/My Drive/GW data/labels_5.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['5', '10', '15', '5', '10', '15', '50', '150', '250', '350', '450', '60', '120', '50', 'UIE', 'CIE', '250', '280', 'Glitch', 'I', 'II', 'III', '10', '30', '30', '60', 'Noise'])
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
  hf = h5py.File('gdrive/My Drive/GW data/mixed_5.h5', 'w')
  hf.create_dataset('mixed_5', data=signal_gw)
  hf.close()

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'

ls gdrive/My\ Drive/GW\ data/

#--------------------------------------------------------
#                           MAIN                    -----
#--------------------------------------------------------
if __name__ == '__main__':
  global next_val
#  #print('\nCollecting noise.....\n data size : '+str(next_val))
#hf= h5py.File('gdrive/My Drive/GW data/noise.h5', 'r')
#group_key = list(hf.keys())[0]
#noise= hf[group_key]
#noise=np.array(noise)
#print(noise.shape,type(noise))
#hf.close()
#noise=noise[:5]
#print(noise.shape)

  hf= h5py.File('gdrive/My Drive/GW data/1164685312_4096.hdf5', 'r')
  group_key = list(hf.keys())
  strain=hf['strain']['Strain'].value
  ts = hf['strain']['Strain'].attrs['Xspacing']
  metaKeys = hf['meta'].keys()
  meta = hf['meta']
  gpsStart = meta['GPSstart'].value
  duration = meta['Duration'].value
  gpsEnd   = gpsStart + duration
  time = np.arange(gpsStart, gpsEnd, ts)

  noise=np.zeros((6,49152))
  for i in range(noise.shape[0]-1):
    numSamples = 4096*12*i
    next_numSamples = 4096*12*(i+1)
    print(numSamples, next_numSamples)
    noise[i] = strain[numSamples:next_numSamples]
  print('noise : ',noise.shape)

  signal_gw=np.zeros((60000,noise.shape[1]))
  next_val=0

  print('\n\nPreparing data..... ')
  train_pipeline(noise) #--------training-------------#
  print('\nPreparing data........100%\n\n')


  k=signal_gw[:next_val,:]
  print(k.shape)
  for e in range(k.shape[0]):
    k[e]=(k[e]-np.mean(k[e] ,axis=0))/np.std(k[e])

  alb=k.reshape((k.shape[0],128,128,3))
  print(alb.shape)
  for i in tqdm(range(alb.shape[0])):
    plt.axis("off")
    plt.imshow((alb[i]* 255).astype(np.uint8), cmap=None, interpolation='nearest')
    plt.savefig('gdrive/My Drive/GW data/Image/'+str(i)+'.png')
