# import the necessary packages
import sys

if sys.platform == "darwin":
    sys.path.insert(0, './macos')

from flask import Flask, render_template, Response
from scam import VideoCamera


camera = VideoCamera()

app = Flask(__name__, static_url_path='', static_folder='templates')
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        #print("POST")
        if request.form.get('Start') == 'Start':
            #print("start here")
            camera.start_button()
        if request.form.get('Stop') == 'Stop':
            camera.stop_button()
    # rendering webpage
    return render_template('index.html')

def gen(camera):
    print("here1")
    while True:
        #get camera frame
        frame = camera.get_frame()
        if frame !=  0:
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    print("video feed: ")
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # defining server ip address and port
    app.run(host='0.0.0.0',port='5000', debug=True)
