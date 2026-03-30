"""
Opens a video and saves it with a new specific frame rate (FPS).
This adjusts the playback speed without discarding or adding frames.
"""

import cv2
import os

def change_video_fps():
    # =========================
    # Configuration
    # =========================
    input_path = r"C:\Users\Sophia\Documents\20260309_Coleta Vinicius\20260309_Coleta Vinicius\video004.avi"
    output_path = r"C:\Users\Sophia\Documents\20260309_Coleta Vinicius\20260309_Coleta Vinicius\video004_corrected.avi"
    
    # Set the desired frame rate here (e.g., 50.0 if 3000 frames should last 60 seconds)
    target_fps = 50.0 

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get original properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original FPS: {original_fps:.2f}")
    print(f"Frame Count: {frame_count}")
    print(f"Target FPS: {target_fps:.2f}")
    print(f"Original Duration: {frame_count / original_fps:.2f}s")
    print(f"New playback duration will be: {frame_count / target_fps:.2f}s")

    # Define codec (I420 is common for raw/uncompressed rPPG videos)
    fourcc = cv2.VideoWriter_fourcc(*'I420') 
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        count += 1
        if count % 500 == 0:
            print(f"Processed {count}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"Success! Fixed video saved to: {output_path}")

if __name__ == "__main__":
    change_video_fps()
