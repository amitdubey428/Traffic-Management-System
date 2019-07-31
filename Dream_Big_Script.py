"""
Description
@file dynamicTrafficManagement.py

The dataset for this model is taken from the Universitetsboulevarden,
Aalborg, Denmark

This file calculate the density of the traffic and set the value of the timer
for each four lane

"""

"""
# Import statements #
"""
import cv2
import numpy as np
from sklearn.externals import joblib
from threading import *
import time

# Reading Reference Image for BackGround Subtraction#
refIm = cv2.imread('refFrame.jpg')
refIm2 = cv2.cvtColor(refIm, cv2.COLOR_BGR2GRAY)


# setting region of the interest#
roi = np.ones(refIm2.shape, "uint8")
cv2.rectangle(roi, (62, 60), (242, 180), 255, -1)

bg = refIm2.copy()
bg = cv2.bitwise_and(bg, roi)


# importing linearRegression model pickle file#
model = joblib.load("model.cpickle")


"""
#Global TIME variable for setting the variable and using in main#
"""
TIME = 0


# timer class for setting the timer running on different thread#
class SetTimer(Thread):
    def run(self):
        final_time = TIME
        print("predicted time is ", final_time)
        for i in range(final_time):
            # delay of one second#
            time.sleep(1)
            print(final_time - i)


# Calculating the frame number from the given time as an argument#
def calcFrame(x, y):
    frame_time = int((x * 60 + y) * 35)
    return frame_time


# Processing the detected frame #
"""
# roi is setted on the frame #
# Then the background subtraction is done #
# Then frame is converted into binary form #
# Then noise is removed by morphological operations #
# Then frame is dilated to properly detect the contours #
# Then contour is drawn and area is calculated #
# From the area the model predict the time # 
# This time is setted to the global variable TIME #
"""

def process(frame):
    vidClone = frame.copy()
    global roi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, roi)
    diff = cv2.absdiff(bg.astype('uint8'), gray)
    # threshold logic#
    thresh = 53
    thresholded = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)[1]
    # Opening logic#
    k = 3
    kernel = np.ones((k, k), "uint8")
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    # dilation logic#
    dilate = 15
    dilated = cv2.dilate(opening, None, iterations=dilate)
    # change to _,contour,_ for latest version#
    contour, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Finding the area of each contour#
    for i in range(len(contour)):

        M = cv2.moments(contour[i])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour[i])
        if area >= 3700:
            cv2.drawContours(vidClone, contour, i, (0, 255, 0), 3)
            arr = np.array([area])
            arr = np.reshape(arr, (-1, 1))
            time = model.predict(arr)
            global TIME
            TIME = int(time)
    # cv2.imshow("vidClone", vidClone)
    return vidClone
    keypress = cv2.waitKey(1) & 0xFF
    # if the user pressed "q", then stop looping
    if keypress == ord('q'):
        return

#Main function for input and output displaying#

if __name__ == "__main__":
    # Capturing Four lanes#
    vid1 = cv2.VideoCapture('latestData.mp4')
    vid2 = cv2.VideoCapture('latestData.mp4')
    vid3 = cv2.VideoCapture('latestData.mp4')
    vid4 = cv2.VideoCapture('latestData.mp4')
    temp = np.zeros(refIm.shape,"uint8")
    timer = temp.copy()

    # setting the video frame for different lanes#
    #For lane1 #
    lane1_start_time = calcFrame(1, 60)
    lane1_end_time = calcFrame(2, 26)
    vid1.set(1, lane1_start_time)
    _, frame1 = vid1.read()

    #For lane2 #
    lane2_start_time = calcFrame(2, 52)
    lane2_end_time = calcFrame(3, 25)
    vid2.set(1, lane2_start_time)
    _, frame2 = vid2.read()

    #For lane3#
    lane3_start_time = calcFrame(6, 56)
    lane3_end_time = calcFrame(7, 26)
    vid3.set(1, lane3_start_time)
    _, frame3 = vid3.read()

    #For lane4#
    lane4_start_time = calcFrame(12, 22)
    lane4_end_time = calcFrame(12, 52)
    vid4.set(1, lane4_start_time)
    _, frame4 = vid4.read()

    # display window. fWin is the final Video#
    st0 = np.hstack((temp, frame1, temp))
    st1 = np.hstack((frame4, timer, frame2))
    st2 = np.hstack((temp, frame3, temp))
    fWin = np.vstack((st0, st1, st2))

    # lane1#
    vid1.set(1, calcFrame(2, 15))
    while vid1.get(1) <= (lane1_end_time):

        ret1, frame1 = vid1.read()
        frame1 = process(frame1)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        fWin = np.vstack((st0, st1, st2))
        if vid1.get(1) == calcFrame(2, 23):
            _t = SetTimer()
            _t.start()
            _t.join(TIME - 6)
        x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
        cv2.putText(fWin, 'Green Window for Lane 1:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.putText(fWin, str(TIME), (x + 10, y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane2#

    while vid2.get(1) <= (lane2_end_time):

        ret1, frame2 = vid2.read()
        frame2 = process(frame2)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        if vid2.get(1) == calcFrame(3, 21):
            _t = SetTimer()
            _t.start()
            _t.join(TIME - 6)
        fWin = np.vstack((st0, st1, st2))
        x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
        cv2.putText(fWin, 'Green Window for Lane 2:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.putText(fWin, str(TIME), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane3#

    while vid3.get(1) <= (lane3_end_time) and TIME:

        ret1, frame3 = vid3.read()
        frame3 = process(frame3)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        if vid3.get(1) == calcFrame(7, 22):
            _t = SetTimer()
            _t.start()
            _t.join(TIME - 6)
        fWin = np.vstack((st0, st1, st2))
        x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
        cv2.putText(fWin, 'Green Window for Lane 3:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.putText(fWin, str(TIME), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        # time.sleep(1)

        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

    # lane4#

    while vid4.get(1) <= (lane4_end_time):

        ret1, frame4 = vid4.read()
        frame4 = process(frame4)
        st0 = np.hstack((temp, frame1, temp))
        st1 = np.hstack((frame4, timer, frame2))
        st2 = np.hstack((temp, frame3, temp))
        if vid4.get(1) == calcFrame(12, 35):
            _t = SetTimer()
            _t.start()
            _t.join(TIME - 6)
        fWin = np.vstack((st0, st1, st2))
        x, y = int(fWin.shape[0] / 2) + 50, int(fWin.shape[1] / 2) - 80
        cv2.putText(fWin, 'Green Window for Lane 4:', (x - 50, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
        cv2.putText(fWin, str(TIME), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imshow("frame", fWin)
        keypress = cv2.waitKey(1) & 0xFF
        # if the user pressed "q", then stop looping
        if keypress == ord('q'):
            break

# destroy all windows#
cv2.destroyAllWindows()
