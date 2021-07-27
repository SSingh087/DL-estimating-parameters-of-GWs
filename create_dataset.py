import numpy as np
from pycbc import waveform, detector
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
        check=np.zeros(noise.shape[1])
                k=0
                for m1 in range(5,16,5):
                  for m2 in range(5,16,5):
                    for d in [50, 250, 450]:
                      for fu in [60,120]:
                        if (m1+m2+d+fl) not in check:
                          check[k]=m1+m2+d+fl
                          hp,hc = waveform.get_td_waveform(approximant=apx[a],
                                            mass1=m1,mass2=m2,
                                            delta_t=1.0/4096,
                                            f_lower=50, f_final=fu, 
                                            distance=d)
                          strain = detector.Detector('H1').project_wave(hp, hc, 0, 0, 1.75) 
                          if len(strain)<=noise.shape[1]:
                            signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
                            pos=np.random.randint(0,noise.shape[1]-len(strain))
                            signal_gw[next_val:next_val+noise.shape[0],pos:pos+len(strain)]+=strain
                            writer = csv.writer(file)
                            for i in range(noise.shape[0]):
                              col=np.zeros(23)
                              # Store 1 in those columns corresponding to injected values
                              if m1 % 5 == .. :
                                col[..] = 1
                                ..
                                ..
                            writer.writerow(col)
                          k=k+1
            #--------update next_val ----------#
                          next_val+=(noise.shape[0])


#------------------------------------------------------------------
#                    ECHOES +TRANSIENT NOISE
#------------------------------------------------------------------
  def echoes(noise):
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for loop in tqdm(range(10)):
        t=np.linspace(0,.3,np.random.randint(noise.shape[1]))
        y1,y2=np.zeros(len(t)),np.zeros(len(t))
        i=0
        for j in range(8):
          for f in [250,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295
              y1[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            pos=np.random.randint(0,noise.shape[1]-len(t))
            signal_gw[next_val:next_val+noise.shape[0]-1,pos:pos+len(t)]+=y1
            next_val+=(noise.shape[0])
            writer = csv.writer(file)
            for i in range(noise.shape[0]):
              col=np.zeros(23)
              # Store 1 in those columns corresponding to injected values
              if m1 % 5 == .. :
                col[..] = 1
                ..
                ..
            writer.writerow(col)

        r=.3
        for j in range(8):
          for f in [250,280]:
            for i in range(len(t)):
              aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
              y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*f*aa)
            pos=np.random.randint(0,noise.shape[1]-len(t))
            signal_gw[next_val:next_val+noise.shape[0]-1,pos:pos+len(t)]+=y2
            next_val+=(noise.shape[0])
            writer = csv.writer(file)          
            for i in range(noise.shape[0]):
              col=np.zeros(23)
              # Store 1 in those columns corresponding to injected value
                ..
                ..
            writer.writerow(col)

#------------------------------------------------------------------
#                    GLITCHES+TRANSIENT NOISE
#------------------------------------------------------------------
  def glitches(noise):
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0],:]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for i in tqdm(['...']): # LIST OF FILES
        loc='gdrive/My Drive/GW data/Glitches/'+i+'.wav'
        rate,data=s.read(loc)
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
    global next_val,signal_gw
    signal_gw[next_val:next_val+noise.shape[0]]=np.copy(noise)
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      val=['..'] # .dat Files
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
            pos=np.random.randint(0,noise.shape[1]-len(new_arr))
            signal_gw[next_val:next_val+noise.shape[0],pos:pos+len(new_arr)]+=new_arr
            writer = csv.writer(file)
            for i in range(noise.shape[0]):
              col=np.zeros(23)
              # Store 1 in those columns corresponding to injected value
                ..
                ..
            writer.writerow(col)


#------------------------------------------------------------------------------
#                             MIXED SIGNALS  BBH/BNS/GW-Echoes     ------------
#------------------------------------------------------------------------------
  def mixed_signals_BHBNSB(noise):
    global next_val,signal_gw

    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      for aab in ['TaylorT1', 'EOBNRv2']:
        check=np.zeros(noise.shape[1])
        k=0
        for m1 in tqdm(range(5,16,5)):
          for m2 in range(5,16,5):
            for d in [50, 250, 450]:
              for fu in [60,120]:
                if (m1+m2+d+fl) not in check:
                  check[k]=m1+m2+d+fl
                  hp,hc = get_td_waveform(approximant=aab,
                                    mass1=m1,mass2=m2,
                                    delta_t=1.0/4096,
                                    f_lower=50, f_final=fu, 
                                    distance=d)
                  strain = detector.Detector('H1').project_wave(hp, hc, 0, 0, 1.75) 
                  if len(strain)<=noise.shape[1]:
                    t=np.linspace(0,.3,np.random.randint(noise.shape[1]))
                    y2=np.zeros(len(t))          
                    r=.3
                    for j in range(3,8):
                      for i in range(len(t)):
                        aa=t[i]-0.0295-j*0.0295-(j*(j+1)/2)*r*0.0295
                        y2[i]+=1.5*10e-21*(-1)**j*(1.5*10e-21*.5/(3+j))*np.exp(-(aa**2)/(2*.006**2))*np.cos(2*np.pi*250*aa)

                        
                      ## REPEAT FOR CIE ##
                      for i in tqdm(['...']): # LIST OF FILES
                        loc='gdrive/My Drive/GW data/Glitches/'+i+'.wav'
                        rate,data=s.read(loc)
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
#                             MIXED SIGNALS CCSNe                 -------------
#------------------------------------------------------------------------------
  def mixed_signals_CCSNe(noise):
    global next_val,signal_gw  
    with open('gdrive/My Drive/GW data/labels_5.csv', 'a', newline='') as file:
      val=['..'] # .dat Files
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
            
            for i in tqdm(['...']): # LIST OF FILES
              loc='gdrive/My Drive/GW data/Glitches/'+i+'.wav'
              rate,data=s.read(loc)
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
