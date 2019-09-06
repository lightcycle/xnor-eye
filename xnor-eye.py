from flask import Flask, jsonify, render_template, Response
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

OUTLINE_COLOR = (255, 0, 0)  # Red
OUTLINE_WIDTH = 5

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

def getserial():
    cpuserial = "0000000000000000"
    try:
        f = open('/proc/cpuinfo','r')
        for line in f:
            if line[0:6]=='Serial':
                cpuserial = line[10:26]
        f.close()
    except:
        cpuserial = "ERROR000000000"
 
    return cpuserial

app = Flask(__name__)
Compress(app)

@app.route('/')
def index():
    result = doInference()
	
    return render_template('index.html', labels = str(result['labels']), imagedata = result['image'], width = input_resolution[0], height = input_resolution[1])

@app.route('/evaluate/')
def evaluate():
    result = doInference()
	
    if result is None:
        return {'error': 'no camera data yet'}, status.HTTP_204_NO_CONTENT
		
    return jsonify(result)

def gen():
    while True:
        result = doInferenceRaw();
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + result['imageRaw'] + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def doInference():
    result = doInferenceRaw()
    return {'labels': result['labels'], 'image': base64.b64encode(result['imageRaw']).decode('ascii'), 'serial': getserial()}
	
def doInferenceRaw():
    cam_buffer = stream.getvalue()
    if len(cam_buffer) != SINGLE_FRAME_SIZE_RGB:
        return None
    model_input = xnornet.Input.rgb_image(input_res[0:2], cam_buffer)
    boxes = model.evaluate(model_input)
	
    labels = []
    for box in boxes:
	    labels.append('(%f,%f,%f,%f)' % (box.rectangle.x, box.rectangle.y, box.rectangle.x + box.rectangle.width, box.rectangle.y + box.rectangle.height))

    image = Image.frombytes("RGB", input_res[0:2], cam_buffer)

    drawer = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for box in boxes:
        top_left = (int(box.rectangle.x * image_width),
                    int(box.rectangle.y * image_height))
        bottom_right = (
            int((box.rectangle.x + box.rectangle.width) * image_width),
            int((box.rectangle.y + box.rectangle.height) * image_height))
        coords = [(top_left[0], top_left[1]),
                  (top_left[0], bottom_right[1]),
                  (bottom_right[0], bottom_right[1]),
                  (bottom_right[0], top_left[1])]
        coords += coords[0]
        drawer.line(coords, fill=OUTLINE_COLOR, width=OUTLINE_WIDTH)

    byte_io = BytesIO()
    image.save(byte_io, 'JPEG', quality=70)

    return {'labels': labels, 'imageRaw': byte_io.getvalue(), 'serial': getserial()}

if __name__ == "__main__":
    app.run(host='0.0.0.0')
