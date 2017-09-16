
from multiprocessing import Queue, Lock
from Queue import Empty
from rocket import Rocket
from threading import Thread
import signal
import time
import cv2
import logging, logging.handlers

from capture import Capture
from server import *

QUEUE_MAXSIZE = 10
PORT = 5000

# Setup logging
logFormatter = logging.Formatter(fmt='%(levelname)-8s %(module)-15s %(asctime)-20s - %(message)s',
	datefmt='%m/%d/%Y %I:%M:%S %p')
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# Initialize and configure threads
app.pre_queue = Queue(maxsize=QUEUE_MAXSIZE)

# Make main process ignore SIGNINT
original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

app.capture_th = Capture(app.pre_queue)

app.capture_th.setDevice("video0")
    
# Launch threads
app.capture_th.start()
logging.info("Threads started.")

# Restore SIGNINT handler
signal.signal(signal.SIGINT, original_sigint_handler)

# Launch server
rocket_server = Rocket(('0.0.0.0', PORT), 'wsgi', {'wsgi_app': app})
app.server_th = Thread(target=rocket_server.start, name='rocket_server')
app.server_th.start()
logging.getLogger("Rocket").setLevel(logging.INFO)
logging.info("Server started.")

try:
    while app.server_th.is_alive():
        app.server_th.join(1)
except (KeyboardInterrupt, SystemExit):
    rocket_server.stop()
    logging.info("Server stopped.")

app.capture_th.stop()
app.capture_th.join()

cv2.destroyAllWindows()
