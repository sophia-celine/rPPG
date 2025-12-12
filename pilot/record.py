#!/usr/bin/env python3

import sys
import os
import cv2
import neoapi
import csv
import argparse
from datetime import datetime

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Record video from a Baumer camera.")
parser.add_argument('--focus', type=str, required=True, help='Focus value.')
parser.add_argument('--aperture', type=float, required=True, help='Aperture value of the camera.')
parser.add_argument('--luminosity', type=float, required=True, help='Luminosity value in roi area.')
args = parser.parse_args()

# --- Recording Parameters ---
result = 0
fps = 15
pixel_format = 'BGR8'
exposure_time = 30000
width = 2000 #2464
height = 2028 #2048
video_writer = 'XVID'
csv_log_file = 'recordings_log.csv'
duration_seconds = 5
focus = args.focus
aperture = args.aperture
lum = args.luminosity

try:
    camera = neoapi.Cam()
    camera.Connect()

    isColor = True
    if camera.f.PixelFormat.GetEnumValueList().IsReadable(pixel_format):
        camera.f.PixelFormat.SetString(pixel_format)
    elif camera.f.PixelFormat.GetEnumValueList().IsReadable('Mono8'):
        camera.f.PixelFormat.SetString('Mono8')
        isColor = False
    else:
        print('no supported pixelformat')
        sys.exit(0)

    camera.f.ExposureTime.Set(exposure_time)
    camera.f.AcquisitionFrameRateEnable.value = True
    camera.f.AcquisitionFrameRate.value = fps
    camera.f.Width.value = width
    camera.f.Height.value = height

    now = datetime.now()
    filename = now.strftime("%H-%M-%S-%d-%m-%Y") + ".avi"

    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*video_writer), fps,
                            (camera.f.Width.value, camera.f.Height.value), isColor)
    print('Recording...')
    for cnt in range(0, duration_seconds * fps):
        print(camera.f.AcquisitionFrameRate.value)
        img = camera.GetImage().GetNPArray()
        video.write(img)

    video.release()
    print(f"Recording complete. Video saved as '{filename}'")

    recording_data = {
        'timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
        'video_filename': filename,
        'fps': fps,
        'pixel_format': pixel_format,
        'exposure_time': exposure_time,
        'width': width,
        'height': height,
        'video_writer': video_writer,
        'duration_seconds': duration_seconds,
        'focus': focus,
        'aperture': aperture,
        'luminosity': lum
    }

    file_exists = os.path.isfile(csv_log_file)
    with open(csv_log_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=recording_data.keys())
        if not file_exists:
            writer.writeheader()  # Write header if file is new
        writer.writerow(recording_data)
    print(f"Metadata saved to '{csv_log_file}'")

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc)
    result = 1

sys.exit(result)
