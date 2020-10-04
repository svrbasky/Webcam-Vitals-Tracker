"""
Webcam Stream:
This class will access webcam and display video on webpage.

"""
from imutils.video import WebcamVideoStream
import cv2


class webCam(object):
    # Open stream
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()

    # Close stream
    def __del__(self):
        self.stream.stop()

    # Read Frame
    def get_frame(self):
        image = self.stream.read()

        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        return data
