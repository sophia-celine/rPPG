import argparse
import os
import cv2


def overlay_text(frame, lines, pos=(10, 20), color=(0, 255, 0)):
	x, y = pos
	for i, line in enumerate(lines):
		cv2.putText(frame, line, (x, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def human_filename(path):
	base = os.path.basename(path)
	name, ext = os.path.splitext(base)
	return name


def write_cut_video(input_path, start_frame, output_path=None, codec='mp4v'):
	cap = cv2.VideoCapture(input_path)
	if not cap.isOpened():
		raise RuntimeError(f"Can't open input: {input_path}")

	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

	if output_path is None:
		dirname = os.path.dirname(input_path)
		name = human_filename(input_path)
		output_path = os.path.join(dirname, f"{name}_cut_{start_frame:06d}.mp4")

	fourcc = cv2.VideoWriter_fourcc(*codec)
	writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	idx = start_frame
	print(f"Writing from frame {start_frame} to end into: {output_path}")
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		writer.write(frame)
		idx += 1

	writer.release()
	cap.release()
	print(f"Saved cut video: {output_path}")
	return output_path


def interactive_cut(input_path, output_path=None):
	cap = cv2.VideoCapture(input_path)
	if not cap.isOpened():
		print('Error: cannot open video file')
		return

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

	current = 0
	start_frame = None

	win = 'Cut Video' 
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)

	instructions = [
		"Controls: a - prev, d - next, s - select start",
		"w - write cut video, q - quit"
	]

	while True:
		cap.set(cv2.CAP_PROP_POS_FRAMES, current)
		ret, frame = cap.read()
		if not ret:
			# if we can't read the frame, stop
			print('End of video or read error')
			break

		display = frame.copy()
		lines = [f"Frame: {current+1}/{total_frames}  FPS:{fps:.2f}"]
		if start_frame is not None:
			lines.append(f"Selected start: {start_frame+1}")
		else:
			lines.append("Selected start: -")
		lines.extend(instructions)
		overlay_text(display, lines)

		if start_frame is not None and current >= start_frame:
			# visual hint: draw a red rectangle when we're in the selected region
			cv2.rectangle(display, (0, 0), (width-1, height-1), (0, 0, 255), 4)

		cv2.imshow(win, display)
		key = cv2.waitKey(0) & 0xFF

		if key == ord('d') or key == 83:  # next
			if current < total_frames - 1:
				current += 1
		elif key == ord('a') or key == 81:  # prev
			if current > 0:
				current -= 1
		elif key == ord('s'):
			start_frame = current
			print(f"Selected start frame: {start_frame}")
		elif key == ord('w'):
			if start_frame is None:
				# allow saving from current if no start selected
				print('No start selected; using current frame as start')
				start_frame = current
			try:
				out = write_cut_video(input_path, start_frame, output_path)
				print('Done:', out)
			except Exception as e:
				print('Error while writing cut video:', e)
		elif key == ord('q'):
			break
		else:
			# any other key: skip forward by 1
			if key != 255:
				# unknown key pressed; ignore
				pass

	cap.release()
	cv2.destroyAllWindows()


def parse_args():
	p = argparse.ArgumentParser(description='Interactively cut a video starting at a selected frame')
	p.add_argument('input', help='Input video path')
	p.add_argument('--output', '-o', help='Output video path (optional)')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	if not os.path.exists(args.input):
		print('Input file not found:', args.input)
	else:
		interactive_cut(args.input, args.output)

