#!/usr/bin/env python3

"""Record video from the camera and save it to a file."""

import sys
import cv2
import neoapi

# Camera configuration variables
OUTPUT_FILEPATH = 'imgsize2.avi'
ROI_IMAGE_FILEPATH = 'initial_frame.png'
CODEC = 'XVID'
FRAME_RATE = 50
FRAME_WIDTH =  1056  # max is 2464
FRAME_HEIGHT = 1076  # max is 2048
FRAME_OFFSET_X = 704  # horizontal offset from the sensor origin, in pixels
FRAME_OFFSET_Y = 486  # vertical offset from the sensor origin, in pixels
# FRAME_WIDTH =  2464
# FRAME_HEIGHT = 2048
# FRAME_OFFSET_X = 0  # horizontal offset from the sensor origin, in pixels
# FRAME_OFFSET_Y = 0  # vertical offset from the sensor origin, in pixels
EXPOSURE_TIME_US = 19500
PIXEL_FORMAT = 'BGR8'
RECORD_DURATION_SEC = 5
WINDOW_TITLE = 'Press ESC to stop recording'
GAIN = 1
SAVE_ROI = True


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
        # camera.SetFeature("Gain", GAIN)

        camera.f.Gain.value = GAIN

        camera.f.Width.Set(FRAME_WIDTH)
        camera.f.Height.Set(FRAME_HEIGHT)
        camera.f.OffsetX.Set(FRAME_OFFSET_X)
        camera.f.OffsetY.Set(FRAME_OFFSET_Y)

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

            if SAVE_ROI and frame_index == 0:
                cv2.imwrite(ROI_IMAGE_FILEPATH, image)
                print(f'Saved initial frame to {ROI_IMAGE_FILEPATH}')

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
