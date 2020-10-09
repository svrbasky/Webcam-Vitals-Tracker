import pyrealsense2 as rs
import cv2
import dlib
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter, detrend
import time
import threading
import math

class VideoCamera(object):

    def ext_frame(self):
        while True:
            if self.stop_thread:
                break
            else:
                if (self.infrared_image is not None) and (self.color_image is not None) :
                    irf = cv2.resize(cv2.equalizeHist(infrared_image),(384, 288))
                    cf = cv2.resize(cv2.equalizeHist(color_image[:,:,2]), (384,288))
                    faces = self.fd(irf)
                    if len(faces) > 0:
                        shape = self.sd(irf, faces[0])
                        print('faces: ', len(faces))
                        lr = np.mean(irf[int(shape.part(17).x):int(shape.part(31).x),int(shape.part(28).y):int(shape.part(48).y)]) # Left Cheek IR image
                        hr = np.mean(irf[int(shape.part(18).x):int(shape.part(25).x),int(shape.part(69).y):int((shape.part(25).y + shape.part(1).y)/2 )]) # Forehead
                        rr = np.mean(irf[int(shape.part(35).x):int(shape.part(26).x),int(shape.part(28).y):int(shape.part(54).y)]) # Right Cheek
                        _avgi = float(lr+hr+rr)

                        lc = np.mean(cf[int(shape.part(17).x+15):int(shape.part(31).x+15),int(shape.part(28).y-7):int(shape.part(48).y-7)]) # Left Cheek Color Image
                        hc = np.mean(cf[int(shape.part(18).x+15):int(shape.part(25).x+15),int(shape.part(69).y-7):int((shape.part(25).y + shape.part(18).y)/2-7)]) # Forehead
                        rc = np.mean(cf[int((shape.part(35).x+15)):int(shape.part(26).x+15),int(shape.part(28).y-7):int(shape.part(54).y-7)]) # Right Cheek
                        _avgc = float(lc+hc+rc)
                        _avgr = float(_avgi)/float(_avgc)
                        if self.pause is False and self.Start is True:
                            if (math.isfinite(_avgc) is True) and (math.isfinite(_avgi) is True) and (math.isfinite(_avgr) is True):
                                if (_avgr != 0) and (_avgi != 0) and (_avgc != 0):
                                    self.tsArray.append(time.time())
                                    self.avgcArray.append(_avgc)
                                    self.avgiArray.append(_avgi)
                                    self.avgrArray.append(_avgr)
                        if self.count is None:
                            self.count = 0
                        else:
                            self.count = self.count+1

                        # Face bounding box
                        self.xx1=shape.part(0).x
                        self.yy1=shape.part(69).y # top corner
                        self.xx2=shape.part(16).x
                        self.yy2=shape.part(8).y

                if self.pause:
                    if self.pcount >= 300:
                        self.pause = False
                        self.pcount = 0
                    else:
                        self.pcount += 1

                if len(self.tsArray) >= 64 and self.pause is False and self.Start is True:
                    ica = FastICA(n_components=2, max_iter=1000)
                    L = len(self.tsArray)
                    self.tsamp = (self.tsArray[-1] - self.tsArray[0])
                    self.fps = float(L)/self.tsamp
                    # Band Pass filter
                    nyq = 0.5 * self.fps
                    low = 0.5/nyq # 0.5 Hz to 3.0 Hz
                    high = 3.0/nyq # 0.5 Hz to 3.0 Hz
                    #print(self.fps, nyq, low, high)
                    b, a = b, a = butter(6, [low, high], btype='band')
                    avgi = lfilter(b, a, self.avgiArray)
                    avgc = lfilter(b, a, self.avgcArray)
                    avgr = lfilter(b, a, self.avgrArray)

                    # Remove Baseline Deviation
                    avgi = detrend(avgi)
                    avgc = detrend(avgc)
                    avgr = detrend(avgr)

                    # ICA Normalization
                    ST = np.c_[avgi,avgc]
                    ST-=ST.mean(axis=0)
                    ST/=ST.std(axis=0)
                    ST_=ica.fit(ST).transform(ST)
        
                    avgi = ST_[:,0] # IR
                    avgc = ST_[:,1] # Color
        
                    xs = np.linspace(self.tsArray[0],self.tsArray[-1],L) # Linear interpolation (L should be multiples of 2^n)
                    if (len(avgi) == len(self.tsArray)) and (len(avgc) == len(self.tsArray)) and (len(avgr) == len(self.tsArray)):
                        yc = np.interp(xs,self.tsArray,avgc)
                        interpC = np.hanning(L) * yc
                        interpC = interpC - np.mean(interpC)
                        rawC = np.abs(np.fft.rfft(interpC))

                        yr = np.interp(xs,self.tsArray,avgr)
                        interpR = np.hanning(L) * yr
                        interpR = interpR - np.mean(interpR)
                        rawR = np.abs(np.fft.rfft(interpR))
        
                        yi = np.interp(xs,self.tsArray,avgi)
                        interpI = np.hanning(L) * yi
                        interpI = interpI - np.mean(interpI)
                        rawI = np.abs(np.fft.rfft(interpI))

                        # Compute Heart Rate
                        freqs = 60*float(self.fps)/L*np.arange(L/2+1)
                        idx = np.where((freqs > 60) & (freqs < 120))
                        prunedI = rawI[idx]
                        pfreqI = freqs[idx]
                        idx1I = np.argmax(prunedI)
                        HR_I = pfreqI[idx1I]

                        prunedC = rawC[idx]
                        pfreqC = freqs[idx]
                        idx1C = np.argmax(prunedC)
                        HR_C = pfreqI[idx1C]

                        self.HR = np.mean([HR_C,HR_I])
                        
                        # SPO2
                        RC = max(rawC)/min(rawC) # prunedC[idx1C]/prunedC[0]
                        RI = max(rawI)/min(rawI) # prunedI[idx1I]/prunedI[0]
                        Ratio = RC/RI # Red/Green
                        self.SPO2 = 125-26*Ratio # Taken from a publication

                        # Respiration Rate
                        idx2 = np.where((freqs > 10) & (freqs < 60))

                        prunedC2 = rawC[idx2]
                        pfreqC2 = freqs[idx2]
                        idx3C = np.argmax(prunedC2)
                        RR_C = pfreqC2[idx3C]

                        prunedI2 = rawI[idx2]
                        pfreqI2 = freqs[idx2]
                        idx3 = np.argmax(prunedI2)
                        RR_I = pfreqI2[idx3]

                        self.RR = np.mean([RR_C,RR_I])

                        # Blood Pressure
                        mNPV_i = abs((max(yi)-min(yi))/np.mean(yi))
                        mNPV_c = abs((max(yc)-min(yc))/np.mean(yc))

                        SBP_i = math.exp(0.414*math.log(HR_I)+0.025*math.log(mNPV_i)+2.765) # These numbers are chosen based on Trial and error
                        DBP_i = math.exp(0.389*math.log(HR_I)-0.006*math.log(mNPV_i)+2.715) # They are different for Systolic and Diastolic

                        SBP_c = math.exp(0.414*math.log(HR_C)+0.025*math.log(mNPV_c)+2.765)
                        DBP_c = math.exp(0.389*math.log(HR_C)-0.006*math.log(mNPV_c)+2.715)

                        self.SBP = np.mean([SBP_i, SBP_c])
                        self.DBP = np.mean([DBP_i,DBP_c])

                        print("- FPS:%0.4f - HR:%0.4f - RR:%0.4f - SPO2:%0.4f - SBP:%0.4f - DBP:%0.4f - Elap.Time:%0.4f s -" % (self.fps, self.HR, self.RR, self.SPO2, self.SBP, self.DBP, self.tsamp))
                        #print(len(self.tsArray),self.tsArray)
                        self.pause = True
                        self.pcount = 0
                        self.tsArray = []
                        self.avgcArray = []
                        self.avgiArray = []
                        self.avgrArray = []

    def new_frame(self):
        while True:
            #if self.stop_thread:
            #    break
            self.frames = self.pipeline.wait_for_frames()
            self.infrared_image = np.asanyarray(self.frames.get_infrared_frame().get_data())
            self.color_image = np.asanyarray(self.frames.get_color_frame().get_data())

            #if self.count is not None:
            #    if self.count == 64:
            #        self.stop_thread = True
            #        self.th1.join()
                    #self.th2.join()

    def start_button(self):
        self.Start = True
        
    def stop_button(self):
        self.Start = False

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.fps = 1/60
        self.tsamp = None
        self.st = time.time()
        self.infrared_image = None
        self.color_image = None
        self.frame = None
        self.count = None
        self.Start = False
        self.stop_thread = False
        self.fd = dlib.get_frontal_face_detector()
        self.sd = dlib.shape_predictor('./shape_predictor_81_face_landmarks.dat')
        self.tsArray=[] # Timestamp
        self.avgrArray=[] # Ratio
        self.avgcArray=[] # (Red color)
        self.avgiArray=[] # (Green color)

        # Face's Bounding Box
        self.xx1=0
        self.xx2=0
        self.yy1=0
        self.yy2=0
        
        # Vitals
        self.RR=None # Resp Rate
        self.HR=None # Heart Rate
        self.SPO2=None # SPO2
        self.SBP=None # Systolic
        self.DBP=None # Diastolic
        
        # Display parameters
        self.img = None # Frame
        self.text = None
        self.pause = False
        self.pcount = 0
        self.profile = self.pipeline.start(self.config)
        self.sensor = self.profile.get_device().first_depth_sensor()
        self.sensor.set_option(rs.option.emitter_enabled, 1)
        self.sensor.set_option(rs.option.gain, 32)

        # Thread 1
        self.th1 = threading.Thread(target=self.ext_frame, args=())
        self.th1.daemon = True # Daemon means run in the background
        self.th1.start()

        # Thread 2
        self.th2 = threading.Thread(target=self.new_frame, args=())
        self.th2.daemon = True
        self.th2.start()

    def __del__(self):
        #releasing camera
        self.pipeline.stop()
        self.stop_thread = True
        self.th1.join()
        self.th2.join()

    def get_frame(self):
        if (self.infrared_image is not None) and (self.color_image is not None) :
            cv2.rectangle(self.color_image,(int((self.xx1+15)/0.6),int((self.yy1-7)/0.6)),(int((self.xx2+15)/0.6),int((self.yy2-7)/0.6)),(0,255,0),2)
            cv2.rectangle(self.infrared_image,(int(self.xx1/0.6),int(self.yy1/0.6)),(int(self.xx2/0.6),int(self.yy2/0.6)),(0,255,0),2)

            self.img = np.concatenate((cv2.cvtColor(self.infrared_image,cv2.COLOR_GRAY2BGR), self.color_image), axis=1)
            self.text = "T.Elp: %0.2f s" % (time.time()-self.st)
            cv2.putText(self.img, self.text,(650,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255),1)

            if self.pause is True:
                self.text = "BP:%0.2f/%0.2f " % (self.SBP, self.DBP)
                cv2.putText(self.img, self.text,(650,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)
                self.text = "T.Algo:%0.2f s" % (self.tsamp)
                cv2.putText(self.img, self.text,(650,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)
                self.text = "FPS:%0.2f" % (self.fps)
                cv2.putText(self.img, self.text,(650,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)
                self.text = "HR:%0.2f bpm" % (self.HR)
                cv2.putText(self.img, self.text,(650,300), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)
                self.text = "RR:%0.2f" % (self.RR)
                cv2.putText(self.img, self.text,(650,350), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)
                self.text = "SPO2:%0.2f" % (self.SPO2)
                cv2.putText(self.img, self.text,(650,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255),2)

            ret, jpeg = cv2.imencode('.jpg', self.img)
            return jpeg.tobytes()
        else:
            return 0
        
