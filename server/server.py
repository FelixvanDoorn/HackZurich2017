import flask
from multiprocessing import Queue, Lock, Manager
from Queue import Empty
import logging
import json

app = flask.Flask(__name__)
app.frames_list = []
app.conf = Manager().dict()

@app.route('/')
def home():
    return flask.render_template('base.html')

def stream_feed():
    while True:
	try:
            frame = app.pre_queue.get(block=True, timeout=1)
        except Empty:
            continue
        while not app.pre_queue.empty():
            app.pre_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return flask.Response(stream_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/commands', methods=['POST'])
def commands():
    throttle =  float(flask.request.form['throttle'])
    angle = float(flask.request.form['angle'])
#    logging.info('Received commands [throttle, angle]: [{t},{a}]'.format(t=throttle, a=angle))
    return json.dumps({'status':'OK'})
#    app.actuation.setCommand([throttle, angle])

#@app.get('/static/:path#.+#')
#def server_static(path):
#    return bottle.static_file(path, root="./")
