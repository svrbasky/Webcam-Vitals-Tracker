import cv2
import numpy as np
import time


# Webcam Video Capture Class
class WebcamVitals:
    """ Capture Region of Interest (Forehead) in each frame, create 3 arrays: B,G,R over 1000 frames. Run algorithm on the frame."""
    def __init__(self):
        # Color Pixel data and relative time
        self.blue = []
        self.green = []
        self.red = []
        self.TimeList = []

    def captureData(forehead, t0):
        """Capture ROI in each frame, compute average Blue, Green and Red pixels for each frame and store in arrays"""
        forehead_blue, forehead_green, forehead_red = cv2.split(forehead)

        # Compute average Blue, Green, Red pixels and store in array
        blue.append(np.mean(forehead_blue))
        green.append(np.mean(forehead_green))
        red.append(np.mean(forehead_red))

        # Compute time and append to list
        ts = time.time() - t0
        TimeList.append(ts)



# Check Webcam Feedback and Call Class to run the algorithm
def main():
    """ Open Webcam, Check video feedback to align user's forehead into the box. Then call class methods to capture data and run algorithm."""
    # Open Camera, Display Video Feedback, Rectangular box
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        # Rectangular Box
        # dims
        h,w,x,y = 96,128,240,80
        # box
        frame2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        # Instructions
        frame3 = np.zeros((480, 640, 3), np.uint8)

        # Add Instructions
        instr = "Please adjust your head to align \n your forehead in the box. \n \n Press 'c' to begin capture \n and compute vitals."
        font = cv2.FONT_HERSHEY_SIMPLEX
        y0, dy = 50, 30
        for i, line in enumerate(instr.split('\n')):
            y1 = y0 + i*dy
            cv2.putText(frame3, line, (50, y1), font, 1, (255, 255, 255), 2)

        # Display Video Feedback
        cv2.imshow("Webcam Feedback", frame2)
        cv2.imshow('Instructions', frame3)

        # Key Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            print("Algorithm Starts.")

            # ADD Class Calls for the Algorithm here
            t0 = time.time()
            forehead = frame[y:y+h, x:x+w, :]
            captureData(forehead, t0)

        elif key == ord('q'):
            print("Bye")
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
