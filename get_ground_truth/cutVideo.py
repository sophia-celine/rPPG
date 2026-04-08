"""
Opens an .avi video and crops it between a start frame and an end frame. 
"""

import cv2

input_path = r"C:\Users\Sophia\Documents\20260309_Coleta Vinicius\20260309_Coleta Vinicius\video004_corrected.avi"
output_path = r"C:\Users\Sophia\Documents\20260309_Coleta Vinicius\20260309_Coleta Vinicius\video004_cropped2.avi"
start_frame = 249 
end_frame = 3000

cap = cv2.VideoCapture(input_path)
print("Old frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

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