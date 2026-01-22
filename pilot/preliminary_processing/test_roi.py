import cv2

# rect = (840, 140, 1000, 220)
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L7-11-12-2025-16-23.avi'
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L7-11-12-2025-16-26.avi'
# video_path = r'C:\Users\Sophia\Documents\UTI-11-12-2025\L9-11-12-2025-16-04.avi'
# video_path = r"C:\Users\Sophia\Documents\UTI-11-12-2025\L8-11-12-2025-16-40.avi"
video_path = r'C:\Users\Sophia\Documents\rPPG\initial_tests\videos\60s.avi'
rect = (1200, 500, 1600, 800)

x1, y1, x2, y2 = rect
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Error opening video file.")

ret, first_frame = cap.read()
if not ret:
    cap.release()
    raise ValueError("Unable to read the first frame from the video.")

cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("First Frame with ROI", first_frame)
print("Displaying first frame with ROI... Press any key to continue.")
cv2.waitKey(0)
cv2.destroyAllWindows()