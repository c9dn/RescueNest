from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import base64
import threading
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# Global variables
current_stream = None
stream_lock = threading.Lock()
stream_active = False


def get_device_type():
    user_agent = request.headers.get('User-Agent', '').lower()
    if 'ipad' in user_agent:
        return 'iPad'
    elif 'macintosh' in user_agent or 'mac' in user_agent:
        return 'Mac'
    return 'Unknown'


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def merge_overlapping_boxes(boxes, weights, iou_threshold=0.3):
    if not boxes:
        return [], []

    boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
    weights = weights.tolist() if isinstance(weights, np.ndarray) else weights

    sorted_pairs = sorted(zip(boxes, weights),
                          key=lambda x: x[1],
                          reverse=True)
    boxes = [box for box, _ in sorted_pairs]
    weights = [weight for _, weight in sorted_pairs]

    keep = []
    keep_weights = []

    while boxes:
        keep.append(boxes[0])
        keep_weights.append(weights[0])

        remaining_boxes = []
        remaining_weights = []

        for i in range(1, len(boxes)):
            if compute_iou(boxes[0], boxes[i]) < iou_threshold:
                remaining_boxes.append(boxes[i])
                remaining_weights.append(weights[i])

        boxes = remaining_boxes
        weights = remaining_weights

    return keep, keep_weights


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/user_a')
def user_a():
    return render_template('user_a.html')


@app.route('/user_b')
def user_b():
    return render_template('user_b.html')


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')
    global stream_active
    stream_active = False


@socketio.on('join')
def handle_join(data):
    room = data['room']
    join_room(room)
    logger.info(f'User joined room: {room}')


@socketio.on('leave')
def handle_leave(data):
    room = data['room']
    leave_room(room)
    logger.info(f'User left room: {room}')


# Try different indices in handle_start_stream()
@socketio.on('start_stream')
def handle_start_stream():
    global current_stream
    with stream_lock:
        if current_stream is None:
            try:
                # Test indices in this order
                for index in [0, 1, -1, 2]:
                    current_stream = cv2.VideoCapture(index)
                    if current_stream.isOpened():
                        print(f"Camera opened at index {index}")
                        break
            except Exception as e:
                print(f"Camera error: {str(e)}")


@socketio.on('stop_stream')
def handle_stop_stream():
    global current_stream, stream_active

    with stream_lock:
        if current_stream is not None:
            stream_active = False
            current_stream.release()
            current_stream = None
            emit('stream_stopped', broadcast=True)
            logger.info('Stream stopped successfully')


def stream_video():
    global current_stream, stream_active

    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        while stream_active and current_stream is not None:
            ret, frame = current_stream.read()

            if not ret:
                logger.error('Error reading frame from camera')
                socketio.emit('stream_error', {'message': 'Error reading frame'})
                break

            try:
                frame = cv2.resize(frame, (640, 480))
                # Corrected OpenCV parameter names (snake_case instead of camelCase)
                boxes, weights = hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),  # Note: Some OpenCV versions use win_stride
                    padding=(4, 4),
                    scale=1.03,
                    hitThreshold=0.2  # Note: Some versions use hit_threshold
                )

                valid_boxes = []
                valid_weights = []

                for (x, y, w, h), weight in zip(boxes, weights):
                    aspect_ratio = float(w) / h
                    if (0.25 < aspect_ratio < 0.8 
                            and w > 80 
                            and h > 80
                            and weight > 0.3):
                        valid_boxes.append([x, y, w, h])
                        valid_weights.append(weight)

                merged_boxes = []
                merged_weights = []
                if valid_boxes:
                    merged_boxes, merged_weights = merge_overlapping_boxes(
                        valid_boxes, valid_weights, iou_threshold=0.3
                    )

                    for box, weight in zip(merged_boxes, merged_weights):
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)
                        cv2.putText(
                            frame, 
                            f'Person: {weight:.2f}',
                            (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,
                            (0, 255, 0), 
                            2
                        )

                status = f"People Detected: {len(merged_boxes)}"
                cv2.putText(
                    frame, 
                    status, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, 
                    (0, 0, 255), 
                    2
                )

                _, buffer = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': img_str}, room='stream')

            except Exception as e:
                logger.error(f'Error processing frame: {str(e)}')

            socketio.sleep(0.033)  # ~30 FPS

    except Exception as e:
        logger.error(f'Stream error: {str(e)}')
    finally:
        with stream_lock:
            if current_stream is not None:
                stream_active = False
                current_stream.release()
                current_stream = None
        socketio.emit('stream_stopped', broadcast=True)


if __name__ == '__main__':
    socketio.run(app, debug=True, port=4040, host='0.0.0.0')
