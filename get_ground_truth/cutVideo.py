"""
Opens an .avi video and crops it between a start frame and an end frame. 
"""

import cv2

input_path = r"C:\Users\Sophia\Videos\Baumer Video Records\VCXU.2-57C\video023.avi"
output_path = r"C:\Users\Sophia\Videos\Baumer Video Records\VCXU.2-57C\video023-cropped.avi"

cap = cv2.VideoCapture(input_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Old frame count:", frame_count)

start_frame = 500 
end_frame = frame_count

if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec
fourcc = cv2.VideoWriter_fourcc(*'I420')  # 'XVID', 'MJPG', 'I420'

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Jump to start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current_frame = start_frame
while current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    current_frame += 1

cap.release()
out.release()

cap2 = cv2.VideoCapture(output_path)
print("New frame count:", int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
cap2.release()