import cv2

def frame_diff(prev_frame, cur_frame, next_frame):
	diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
	diff_frames_2 = cv2.absdiff(cur_frame, next_frame)
	return cv2.bitwise_and(diff_frames_1, diff_frames_2)

def get_frame(cap, scaling_factor):
	_, frame = cap.read()
	frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation = cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	return gray

if __name__=='__main__':
	cap = cv2.VideoCapture(0)
	scaling_factor = 0.5
	prev_frame = get_frame(cap, scaling_factor)
	cur_frame = get_frame(cap, scaling_factor)
	next_frame = get_frame(cap, scaling_factor)
	while True:
		cv2.imshow('Object Movement', frame_diff(prev_frame, cur_frame, next_frame))
		prev_frame = cur_frame
		cur_frame = next_frame
		next_frame = get_frame(cap, scaling_factor)
		key = cv2.waitKey(10)
		if key == 27:
			break
	cv2.destroyAllWindows()