# -*- coding: utf-8 -*- 
# @Time 2020/3/4 0:40
# @Author wcy
import cv2
from flask import Flask, render_template, Response
from camera import Camera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/images/<int:flag>')
def video_feed(flag=0):
    return Response(gen(Camera(flag=flag)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)