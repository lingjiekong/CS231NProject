import cv2
vidcap = cv2.VideoCapture('data/20170507T155114.708081.avi')
success,image = vidcap.read()
count = 0
print_count = 0
success = True
while success:
	success,image = vidcap.read()
	if count%10 == 0:
		print 'Read a new frame: ', success
		cv2.imwrite("frame%d.jpg" % print_count, image)     # save frame as JPEG file
		print_count += 1
	count += 1