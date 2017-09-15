import numpy as np
import cv2


PREFIX = "normal"


WIDTH = 1280
HEIGHT = 720

cap = cv2.VideoCapture(1)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

isRecording = False

# video recorder
fourcc =  cv2.VideoWriter_fourcc(*'XVID')# cv2.VideoWriter_fourcc() does not exist


video_writer = cv2.VideoWriter("../testimages/"+str(PREFIX)+"_vid.avi", fourcc, 20, (WIDTH, HEIGHT))

nbImages = 0
nbVids = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    c = cv2.waitKey(1)
    if 'q' == chr(c & 255):
        break 

    if 'c' == chr(c & 255):
        print("Getting image " + str(nbImages))
        cv2.imwrite( "../testimages/im_"+str(PREFIX)+str(nbImages)+".jpg", frame );
        nbImages += 1;

    if 'o' == chr(c & 255):
        if isRecording == False:
            print("Starting recording!" + str(nbVids))
            video_writer = cv2.VideoWriter("../testimages/vid_"+str(PREFIX)+str(nbVids)+".avi", fourcc, 20, (WIDTH, HEIGHT))
            isRecording = True
    
    if 'p' == chr(c & 255):
        if isRecording == True:
            print("Ending recording!" + str(nbVids))
            isRecording = False
            nbVids += 1
            video_writer.release()

    if isRecording:
        video_writer.write(frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()