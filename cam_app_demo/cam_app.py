from flask import Flask, render_template, Response
import cv2
# # emulated camera
from camera import Camera


app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Webcam
# Cam
@app.route('/cam')
def sign_page():
    return render_template("camera.html")

# Camera


@app.route('/camera', methods=['POST'])
def camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/cam.png", img)

        # return render_template("camera.html",result=)
        time.sleep(0.1)
        return json.dumps({'status': 'OK', 'result': "static/cam.png"})
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cap.release()
    # file="/home/ashish/Downloads/THOUGHT.png"
    # with open(file,'rb') as file:
    #     image=base64.encodebytes(file.read())
    #     print(type(image))
    # return json.dumps({'status': 'OK', 'user': user, 'pass': password});
    return json.dumps({'status': 'OK', 'result': "static/cam.png"})

# Camera Function


def gen(camera):
    while True:
        data = camera.get_frame()

        frame = data[0]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Video Feed


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(Camera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
