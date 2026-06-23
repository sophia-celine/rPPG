#!/usr/bin/env python3

"""Record video from the camera and save it to a file."""

import sys
import cv2
import neoapi

# Camera configuration variables
OUTPUT_FILEPATH = 'outpy.avi'
CODEC = 'XVID'
FRAME_RATE = 50
FRAME_WIDTH = 1056
FRAME_HEIGHT = 1076
EXPOSURE_TIME_US = 19500
PIXEL_FORMAT = 'BGR8'
RECORD_DURATION_SEC = 5
WINDOW_TITLE = 'Press ESC to stop recording'


def main():
    result = 0

    try:
        camera = neoapi.Cam()
        camera.Connect()

        is_color = True
        pixel_format_list = camera.f.PixelFormat.GetEnumValueList()
        if pixel_format_list.IsReadable(PIXEL_FORMAT):
            camera.f.PixelFormat.SetString(PIXEL_FORMAT)
        elif pixel_format_list.IsReadable('Mono8'):
            camera.f.PixelFormat.SetString('Mono8')
            is_color = False
        else:
            print('Unsupported pixel format:', PIXEL_FORMAT)
            return 1

        camera.f.ExposureTime.Set(EXPOSURE_TIME_US)
        camera.f.AcquisitionFrameRateEnable.value = True
        camera.f.AcquisitionFrameRate.value = FRAME_RATE

        camera.f.Width.Set(FRAME_WIDTH)
        camera.f.Height.Set(FRAME_HEIGHT)

        width = int(camera.f.Width.value)
        height = int(camera.f.Height.value)

        video_writer = cv2.VideoWriter(
            OUTPUT_FILEPATH,
            cv2.VideoWriter_fourcc(*CODEC),
            FRAME_RATE,
            (width, height),
            is_color,
        )

        total_frames = int(RECORD_DURATION_SEC * FRAME_RATE)
        for frame_index in range(total_frames):
            image = camera.GetImage().GetNPArray()
            if image is None:
                print('Failed to capture frame', frame_index)
                break

            cv2.imshow(WINDOW_TITLE, image)
            video_writer.write(image)

            if cv2.waitKey(1) == 27:  # ESC key
                break

        video_writer.release()
        cv2.destroyAllWindows()
        camera.Disconnect()

    except (neoapi.NeoException, Exception) as exc:
        print('Error:', exc)
        result = 1

    return result


if __name__ == '__main__':
    sys.exit(main())
