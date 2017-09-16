import flask
from multiprocessing import Queue, Lock, Manager
from Queue import Empty
import logging

app = flask.Flask(__name__)
app.frames_list = []
app.conf = Manager().dict()

@app.route('/')
def home():
    return flask.render_template('index.html')

def stream_feed():
    while True:
	try:
            frame = app.pre_queue.get(block=True, timeout=3)
        except Empty:
            continue
        while not app.pre_queue.empty():
            app.pre_queue.get()
	logging.info('New video stream')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return flask.Response(stream_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.get('/static/:path#.+#')
#def server_static(path):
#    return bottle.static_file(path, root="./")
