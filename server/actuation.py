import serial

class Actuation():

    def __init__(self):
	pass
        # Open Serial Connection to Arduino Board
#        self.ser = serial.Serial('/dev/ttyACM0', 9600)

    def setCommand(self, commands):
	throttle = int(commands[0]*50 + 50)
   	angle = abs(int(commands[1]*50 + 50))
#	if j.get_button(4):
#		self.ser.write(b't')
#		time.sleep(0.011)
#		self.ser.write(struct.pack('>B', throttle))
#		time.sleep(0.011)
#		self.ser.write(struct.pack('>B', angle))
#		time.sleep(0.011)
