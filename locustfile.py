from locust import User, task, between, events
import websocket
import json
import base64
import cv2
import time
import numpy as np


class WSUser(User):
    wait_time = between(1, 2)

    def on_start(self):
        self.ws = websocket.create_connection("ws://localhost:8000/ws/stream")

        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode(".jpg", img)

        self.frame_b64 = base64.b64encode(buffer).decode("utf-8")

    @task
    def send_frame(self):
        payload = {
            "frame": self.frame_b64,
            "baseline_model": "Small"
        }

        start = time.time()

        self.ws.send(json.dumps(payload))
        self.ws.recv()

        latency = (time.time() - start) * 1000

        events.request.fire(
            request_type="WS",
            name="frame_inference",
            response_time=latency,
            response_length=0,
            exception=None,
        )

    def on_stop(self):
        self.ws.close()