#!/usr/bin/env python3
"""
This is an example to show case about xnornet surveillance use cases.
The example needs to work with a person classification/detection model.
"""
import argparse
import os
import sys

if sys.version_info[0] < 3:
    sys.exit("This sample requires Python 3. Please install Python 3!")

try:
    from PIL import Image
    from PIL import ImageDraw
except ImportError as e:
    print(e)
    sys.exit("Requires PIL module. "
             "Please install it with pip:\n\n"
             "   pip3 install pillow\n"
             "(drop the --user if you are using a virtualenv)")

try:
    import picamera
except ImportError:
    sys.exit("Requires picamera module. "
             "Please install it with pip:\n\n"
             "   pip3 install picamera\n"
             "(drop the --user if you are using a virtualenv)")

try:
    import xnornet
except ImportError:
    sys.exit("The xnornet wheel is not installed.  "
             "Please install it with pip:\n\n"
             "    python3 -m pip install --user xnornet-<...>.whl\n\n"
             "(drop the --user if you are using a virtualenv)")


def _make_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--input_resolution", action='store', nargs=2, type=int,
        default=(1280, 720),
        help="Input Resolution of the camera, which is also the resolution of "
        "the final saved image.")
    parser.add_argument(
        "--camera_frame_rate", action='store', type=int, default=15,
        help="Adjust the framerate of the camera. 0 indicates a dynamic range "
             "of framerate.")
    parser.add_argument(
        "--camera_brightness", action='store', type=int, default=65,
        help="Adjust the brightness of the camera. Range from 0 to 100.")
    parser.add_argument(
        "--camera_shutter_speed", action='store', type=int, default=1500,
        help="Adjust the shutter speed of the camera in microseconds. "
        "0 means auto shutter speed."
        "https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.shutter_speed"
    )
    parser.add_argument(
        "--camera_video_stablization", action='store_true',
        help="Whether to turn on the video stablization, video "
        "stablization improves video during motion.")
    return parser


def main(args=None):
    parser = _make_argument_parser()
    args = parser.parse_args(args)

    # Reconstruct the input resolution to include color channel
    input_res = (args.input_resolution[0], args.input_resolution[1], 3)
    SINGLE_FRAME_SIZE_RGB = input_res[0] * input_res[1] * input_res[2]

    # Initialize the camera, set the resolution and framerate
    try:
        camera = picamera.PiCamera()
    except picamera.exc.PiCameraMMALError:
        print("\nPiCamera failed to open, do you have another task using it "
              "in the background? Is your camera connected correctly?\n")
        sys.exit("Connect your camera and kill other tasks using it to run "
                 "this sample.")

    # Initialize the buffer for picamera to hold the frame
    # https://picamera.readthedocs.io/en/release-1.13/api_streams.html?highlight=PiCameraCircularIO
    stream = picamera.PiCameraCircularIO(camera, size=SINGLE_FRAME_SIZE_RGB)
    # All essential camera settings
    camera.resolution = input_res[0:2]
    camera.framerate = args.camera_frame_rate
    camera.brightness = args.camera_brightness
    camera.shutter_speed = args.camera_shutter_speed
    camera.video_stabilization = args.camera_video_stablization

    # Record to the internal CircularIO
    camera.start_recording(stream, format="rgb")
    # Load model
    model = xnornet.Model.load_built_in()

    while True:
        # Get the frame from the CircularIO buffer.
        cam_buffer = stream.getvalue()
        # The camera has not written anything to the CircularIO yet
        # Thus no frame is been captured
        if len(cam_buffer) != SINGLE_FRAME_SIZE_RGB:
            continue
        # Passing corresponding RGB
        model_input = xnornet.Input.rgb_image(input_res[0:2], cam_buffer)
        # Evaluate
        results = model.evaluate(model_input)

        print(results);

    print("Cleaning up...")
    camera.stop_recording()
    camera.close()


if __name__ == "__main__":
    main()
