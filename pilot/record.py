#!/usr/bin/env python3

import sys
import cv2
import neoapi

filename = 'output.avi'
result = 0
fps = 25
pixel_format = 'BGR8'
exporsure_time = 36000
width = 2464
height = 2048
video_writer = 'XVID'

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

    camera.f.ExposureTime.Set(exporsure_time)
    camera.f.AcquisitionFrameRateEnable.value = True
    camera.f.AcquisitionFrameRate.value = fps
    camera.f.Width.value = width
    camera.f.Height.value = height

    # video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'MJPG'), 10,
    # video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10,
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*video_writer), fps,
                            (camera.f.Width.value, camera.f.Height.value), isColor)
    print('Recording...')
    for cnt in range(0, 125):
        img = camera.GetImage().GetNPArray()
        video.write(img)
    video.release()
    print('Recording complete')

except (neoapi.NeoException, Exception) as exc:
    print('error: ', exc)
    result = 1

sys.exit(result)


