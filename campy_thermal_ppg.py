import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter, detrend
import cv2
from pylepton import Lepton
import time
import threading
import datetime

global tsArray, tempArray, RR, HR, stop_thread
tsArray=[]
tempArray=[]
HR=None
RR=None
stop_thread = False

def capture(device="/dev/spidev0.0"):
    global tsArray, tempArray
    # Image capture and Normalisation
    with Lepton(device) as l:
        a,_ = l.capture()
    cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX) # extend contrast
    np.right_shift(a, 8, a) # fit data into 8 bits
    tsArray.append(time.time())
    tempArray.append(np.uint8(a).mean())
    return np.uint8(a)

def process_frame():
    global tsArray, tempArray, HR, RR, stop_thread
    while True:
        if stop_thread:
            break

        if len(tsArray) >= 128:
            ica = FastICA(n_components=1, max_iter=1500)
            L = len(tsArray)
            fps = float(L)/(tsArray[-1] - tsArray[0])
        
            #5th order Butterworth BPF [0.4 - 3 Hz] (frequencies taken after considering nyquist criteria)
            nyq = 0.5 * fps #nyquist criteria
            low = 0.4/nyq   
            high = 3.0/nyq
            b, a = b, a = butter(5, [low, high], btype='band')
            temp_bpf = lfilter(b, a, tempArray) # filtered array
        
            #detrend the filtered array to remove motion baseline drift
            temp_detrend = detrend(temp_bpf)
        
            #extract PPG signal using FastICA
            ST = np.c_[temp_detrend]
            ST-=ST.mean()
            ST/=ST.std()
            ST_=ica.fit(ST).transform(ST)
            temp_ica = ST_[:,0]
        
            #Compute HR & RR
            xs = np.linspace(tsArray[0],tsArray[-1],L)
            if (len(temp_ica) == len(tsArray)):
                interpC = np.interp(xs,tsArray,temp_ica) #linear interpoation of data
                interpC = np.hanning(L) * interpC #Apply hanning window
                interpC = interpC - np.mean(interpC)
                rawC = np.abs(np.fft.rfft(interpC)) #FFT
        
                freqs = 60*float(fps)/L*np.arange(L/2+1)
            
                #Compute HR
                idx = np.where((freqs > 60) & (freqs < 180))
                prunedC = rawC[idx]
                pfreqC = freqs[idx]
                idx1C = np.argmax(prunedC)
                HR = pfreqC[idx1C]
        
                #Compute RR
                idx2 = np.where((freqs > 24) & (freqs < 60))
                prunedC2 = rawC[idx2]
                pfreqC2 = freqs[idx2]
                idx3 = np.argmax(prunedC2)
                RR = pfreqC2[idx3]
        
                #Remove used data points
                tsArray.pop(0)
                tempArray.pop(0)
    
def main():
    global HR, RR, stop_thread
    print("IR Vitals App Started")
    
    device="/dev/spidev0.0" #spidev0.0 is the default port of the Lepton Camera on the RPi
    fDelay = 1/8 #8Hz is the max frame rate of Lepton cameras
    cv2.namedWindow('Thermal Window', cv2.WINDOW_AUTOSIZE)
    
    th = threading.Thread(target=process_frame, args=())
    th.daemon = True
    th.start()
    st = time.time()
    
    try:
        while True:
            img = capture(device) # capture image
            
            text = "T.Elp: %0.2f s" % (time.time()-st)
            cv2.putText(img, text,(0,0), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0, 0, 255),0.5)
            
            if HR is not None:
                text = "T.Algo:%0.2f s" % (tsArray[-1]-tsArray[0])
                cv2.putText(img, text,(0,5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0, 0, 255),0.8)
                text = "FPS:%0.2f" % (len(tsArray)/(tsArray[-1]-tsArray[0]))
                cv2.putText(img, text,(0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0, 0, 255),0.8)
                text = "HR:%0.2f bpm" % (HR)
                cv2.putText(img, text,(0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0, 0, 255),0.8)
                text = "RR:%0.2f" % (RR)
                cv2.putText(img, text,(0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.2,(0, 0, 255),0.8)
                
            cv2.imshow('Infrared Example', img)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                stop_thread = True
                break
            time.sleep(fDelay) # Delay
            
    finally:
        th.join()
        cv2.destroyAllWindows()

if __name__== "__main__":
    main()