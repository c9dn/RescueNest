# main.py
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import base64
import numpy as np
import logging
import io

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user_a')
def user_a():
    return render_template('user_a.html')

@app.route('/user_b')
def user_b():
    return render_template('user_b.html')

@app.route('/user_c')
def user_c():
    return render_template('user_c.html')

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('join')
def handle_join(data):
    room = data['room']
    user_type = data['userType']
    join_room(room)
    logger.info(f'{user_type} joined room: {room}')

@socketio.on('leave')
def handle_leave(data):
    room = data['room']
    leave_room(room)
    logger.info(f'User left room: {room}')

@socketio.on('message')
def handle_message(data):
    room = data['room']
    message = data['message']
    user_type = data['userType']
    # Emit the message with user information
    emit('message', {'message': message, 'userType': user_type}, room=room)

@socketio.on('video_frame')
def handle_video_frame(frame_data):
    try:
        # Convert the received frame data to an image
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the frame with HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(
            frame, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.03
        )

        # Draw rectangles around detected people
        for (x, y, w, h), weight in zip(boxes, weights):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'Person: {weight:.2f}',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Add detection count
        status = f"People Detected: {len(boxes)}"
        cv2.putText(
            frame,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Convert processed frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Emit the processed frame to all clients in the stream room
        emit('video_frame', {'image': img_str}, room='stream')
    except Exception as e:
        logger.error(f'Error processing video frame: {str(e)}')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=4040, host='0.0.0.0')
