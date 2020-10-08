import numpy as np
import pandas as pd
import cv2

import time
import datetime
import csv


class WebcamFrame:
    newFrame = []       # New Frame captured from Web camera. Frame is a 3D list 640x480x3(BGR)
    FrameId = ""        # Frame ID is the frame number used as suffix for image names
    TimeList = []       # List of TimeStamps
    TempList = []       # List of average Temperature per Frame
    bluesList = []      # List average value of Blue pixels in ROI of frame
    greensList = []     # List average value of Green pixels in ROI of frame
    redsList = []       # List average value of Red pixels in ROI of frame


    def __init__(self):
        # Contains current frame (Single frame (2D array) at a time)
        self.newFrame = []
        self.TimeList = []  # Full Arrays (useful after full execution)
        self.TempList = []  # Full Arrays (useful after full execution)
        self.FrameId = 1  # Contains current frame ID (Single)

        self.bluesList = []
        self.greensList = []
        self.redsList = []

    # MODIFY THIS HEAVILY
    def captureFrame(self):
        with cv2.VideoCapture(0) as cap:
        ret, frame = cap.read()

        # Draw a rectangle on forehead
        h = 96
        w = 128
        x = 240
        y = 80
        forehead = frame[y:y+h, x:x+w, :] # Frame
        forehead_blue, forehead_green, forehead_red = cv2.split(forehead)
        # forehead_blue = frame[y:y+h, x:x+w, 0] # Blue
        # forehead_green = frame[y:y+h, x:x+w, 1] # Green
        # forehead_red = frame[y:y+h, x:x+w, 2] # Red

        frame2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # Display the resulting frame
        cv2.imshow('frame', frame2)

        # cv2.normalize(a, a, 0, 65535, cv2.NORM_MINMAX)  # extend contrast
        # np.right_shift(a, 8, a)  # fit data into 8 bits

        # Adjust Camera to Align Forehead in ROI Box
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            forehead = frame[y:y+h, x:x+w, :]
            forehead_blue, forehead_green, forehead_red = cv2.split(forehead)

            avg_blue = np.mean(forehead_blue)
            avg_green = np.mean(forehead_green)
            avg_red = np.mean(forehead_red)

            bluesList.append(avg_blue)
            greensList.append(avg_green)
            redsList.append(avg_red)


        cv2.imwrite("capture"+str(self.FrameId) +
                    ".png", np.uint8(frame))  # write it!
        self.FrameId = self.FrameId+1
        self.newFrame = frame


    def processData(self, t0):
        if(len(self.newFrame) != 0):
            self.TempList.append(np.mean(self.newFrame))
            #ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3] # selects the milli seconds and drops the microseconds from time.
            ts = time.time() - t0  # computes relative time
            self.TimeList.append(ts)
            self.newFrame = []


def fftTransform(tempList, timeList):  # This is global fn.
    if(len(tempList) > 150):
        L = len(tempList)
        # Sampling rate = length of samples/[Time between first and last samples]
        fps = float(L) / (timeList[-1] - timeList[0])

        # Create Evenly spaced TimeStamps
        even_times = np.linspace(timeList[0], timeList[-1], L)
        # Linearly interpolate the data according to new TimeStamps
        interpolated = np.interp(even_times, timeList, tempList)
        # Avg of the above(line #46) "interpolated" is body Temperature

        # Apply Hamming window to signal (elementwise multiplication)
        interpolated = np.hamming(L) * interpolated
        interpolated = interpolated - \
            np.mean(interpolated)  # normalising the results
        # Now we have only non DC components left after normalise

        raw = np.fft.rfft(interpolated)  # Compute FFT of interpolated
        # phase = np.angle(raw)
        mag = np.abs(raw)  # Obtaining magnitude of complex FFT
        freqs = (float(fps) / L) * np.arange((L/2) + 1)  # Obtain Freqs in Hz

        # freqs = 60. * freqs # convert freqs into bpm
        # Obtaining Heart rate
        # Find indices where freqs in range (1,4) Hz
        idx = np.where((freqs > 1) & (freqs < 4))

        pruned = mag[idx]  # Prune the magnitudes
        pfreq = freqs[idx]  # Prune the Freqs

        idx2 = np.argmax(pruned)  # Obtain index for max magnitude
        # Beats Per Minute is Heart Rate corresponding to the max magnitude
        heart_rate_bpm = pfreq[idx2] * 60

        # Obtaining Respirotion rate
        # Find indices where freqs in range (0.5,1] Hz
        idx3 = np.where((freqs > 0.5) & (freqs <= 1))

        pruned_RR = mag[idx3]
        pfreq_RR = freqs[idx3]

        idx4 = np.argmax(pruned_RR)
        respiration_rate_pm = pfreq_RR[idx4] * 60

        return heart_rate_bpm, respiration_rate_pm
    else:
        return None


def main():
    # Capture Webcam images
    l = WebcamFrame()
    I = []
    t0 = time.time()
    for i in range(1, 301):
        l.captureFrame()
        l.processData(t0)
        time.sleep(0.1)  # Delay
        I.append(i)  # Frame number

    # Perform FFT to compute Heart Rate in Beats per Minute
    Heart_rate, Resp_rate = fftTransform(l.TempList, l.TimeList)
    if((Heart_rate != None) and (Resp_rate != None)):
        print("Heart Rate: %lf" % Heart_rate)
        print("Respiratory Rate: %lf" % Resp_rate)

    # Print Total Number of Frames
    TotalFrames = "Total Number of Frames = %d" % l.FrameId
    print(TotalFrames)  # Print entire frame IDs to debug

    # Create Pandas DataFrame
    sheet = {'Frame Number': I, 'Frame TimeStamp': l.TimeList,
             'Average Frame Temperature': l.TempList}
    data = pd.DataFrame(
        sheet, columns=['Frame Number', 'Frame TimeStamp', 'Average Frame Temperature'])
    print(data.head())

    # Save Timestamps and number of Frames as CSV file
    data.to_csv(r'Capture_Meta.csv', index=True, header=True)


if __name__ == "__main__":
  main()
