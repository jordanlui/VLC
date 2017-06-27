# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py
# This script is configured to 

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Parameters for operation
camera_address = 1 # 1 for the USB webcam, 0 for the onboard webcam

# define the lower and upper boundaries of the "green"
# ball in the HSV (RGB??) color space, then initialize the
# list of tracked points

# HSV Values for green bottle cap. Optimized for the lamp lighting
# greenLower = (38,70,61)
# greenUpper = (112,250,217)

# HSV Values for table covered with white poster. 20170627
greenLower = (49,80,30)
greenUpper = (107,255,94)

# Bounds for the unlit table are below, but will be less accurate
# greenLower = (66,91,23)	
# greenUpper = (121,211,169)	
pts = deque(maxlen=args["buffer"])
#Green Egg Works better than the Pink one

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(camera_address)

# otherwise, grab a reference to the video file
else:
	camera = cv2.VideoCapture(args["video"])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
size = (640,480)
video = cv2.VideoWriter('output.avi',fourcc, 30.0, size)
# Define and Open coordinates text file
text_file = open("OutputTest.txt", "w")
text_file.close()


time.sleep(2.5)

# keep looping
while True:
	
	# grab the current frame
	(grabbed, frame) = camera.read()
	text_file = open("OutputTest.txt", "a")

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	#Flip the frame
	# Try disabling the flipping, so we get less confused. 20170627
	# frame = cv2.flip(frame, 1)

	# write the frame
	#video.write(frame)
	
	# resize the frame, blur it, and convert it to the HSV
	# color space
	#frame = imutils.resize(frame, width=600) #This line of code seems to prevent the video.write from working later on
	
	# blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# write the frame
	#video.write(frame)
	
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		print((x, y))
		text_file.write("{X: %.3f, Y: %.3f} \n" % (x, y))

		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in xrange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue

		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	# write the frame
	video.write(frame)

	text_file.close()
	time.sleep(0.07725)
	
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
video.release()
cv2.destroyAllWindows()
text_file.close()