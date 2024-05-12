from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from frame_processor import process_frame

app = Flask(__name__, static_url_path='', static_folder='static')
socketio = SocketIO(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


printed_image = 0


@socketio.on('image')
def image(data_image):
    global printed_image
    # Decode the received image data
    decoded_data = base64.b64decode(data_image)
    np_data = np.frombuffer(decoded_data, np.uint8)
    if printed_image == 0:
        # Save the image to the disk
        with open('image.jpg', 'wb') as file:
            file.write(decoded_data)
        printed_image = 1
    img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    try:
        processed_frame = process_frame(img)
    except Exception as e:
        print(e)
        # Emit the same image back if an error occurs during processing
        emit('response_back', data_image)
        return

    # Encode the processed frame to base64 string
    _, imgencode = cv2.imencode('.jpg', processed_frame)
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # Emit the processed frame back
    emit('response_back', stringData)


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1')
