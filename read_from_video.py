import cv2
import numpy as np

def select_white(image):
    color_select  = np.copy(image)
    red_threshold = 190
    blue_threshold = 190
    green_threshold = 190
    rgb_threshold = [red_threshold,blue_threshold,green_threshold]
    # Identify pixels below the threshold
    thresholds = (image[:,:,0] < rgb_threshold[0]) \
            |(image[:,:,1] < rgb_threshold[1]) \
            |(image[:,:,2] < rgb_threshold[2])

    color_select[thresholds] = [0,0,0]
    
    return color_select

cap = cv2.VideoCapture('solidYellowLeft.mp4')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name',select_white(frame))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cap.destroyAllWindows()
