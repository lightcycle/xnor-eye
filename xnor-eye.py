from flask import Flask
from flask import jsonify
from flask_api import status
from flask_compress import Compress
from io import BytesIO
import os
import sys
import base64

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

input_resolution = (1280, 720)
camera_frame_rate = 15
camera_brightness = 65
camera_shutter_speed = 1500
camera_video_stablization = True

input_res = (input_resolution[0], input_resolution[1], 3)
SINGLE_FRAME_SIZE_RGB = input_res[0] * input_res[1] * input_res[2]

try:
	camera = picamera.PiCamera()
except picamera.exc.PiCameraMMALError:
	print("\nPiCamera failed to open, do you have another task using it "
		  "in the background? Is your camera connected correctly?\n")
	sys.exit("Connect your camera and kill other tasks using it to run "
			 "this sample.")

stream = picamera.PiCameraCircularIO(camera, size=SINGLE_FRAME_SIZE_RGB)
camera.start_recording(stream, format="rgb")
model = xnornet.Model.load_built_in()

app = Flask(__name__)
Compress(app)

@app.route('/evaluate/')
def evaluate():
    cam_buffer = stream.getvalue()
    if len(cam_buffer) != SINGLE_FRAME_SIZE_RGB:
        return {'error': 'no camera data yet'}, status.HTTP_204_NO_CONTENT
    model_input = xnornet.Input.rgb_image(input_res[0:2], cam_buffer)
    results = model.evaluate(model_input)

    image = Image.frombytes("RGB", input_res[0:2], cam_buffer)
    byte_io = BytesIO()
    image.save(byte_io, 'JPEG', quality=70)

    return jsonify({'results': results, 'image': base64.b64encode(byte_io.getvalue()).decode('ascii')})
