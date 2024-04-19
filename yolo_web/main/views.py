import random
import cv2
import numpy as np
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from ultralytics import YOLO
from .sort import Sort


class ObjectDetector:
    def __init__(self):
        self.tracker = Sort()
        self.object_id_map = {}  # Словарь для отображения трекеров на идентификаторы объектов
        self.next_object_id = 0
        # Read class list once and store it
        with open("../utils/coco.txt", "r") as f:
            self.class_list = f.read().split("\n")
        self.detection_colors = self.generate_colors(len(self.class_list))
        self.model = YOLO("weights/yolov8n.pt", "v8")
        self.frame_width = 640
        self.frame_height = 480

    def generate_colors(self, num_classes):
        colors = []
        for _ in range(num_classes):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return colors

    def video_detection(self):
        cap = cv2.VideoCapture(0) 

        if not cap.isOpened():
            print("Cannot open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                detections = self.model.predict(source=[frame], conf=0.45, save=False)[0]
                if detections:
                    bbs = []
                    for box in detections.boxes:
                        cls_id = int(box.cls.numpy()[0])
                        conf = box.conf.numpy()[0]
                        bb = box.xyxy.numpy()[0]

                        # Получаем объектный идентификатор
                        object_id = self.object_id_map.get(cls_id, self.next_object_id)
                        if cls_id not in self.object_id_map:
                            self.object_id_map[cls_id] = self.next_object_id
                            self.next_object_id += 1

                        bbs.append([bb[0], bb[1], bb[2], bb[3], conf, cls_id, object_id])

                        cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), self.detection_colors[cls_id], 3)
                        cv2.putText(frame, f"{self.class_list[cls_id]} {conf:.3f}% ID:{object_id}", (int(bb[0]), int(bb[1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                    trackers = self.tracker.update(np.array(bbs))
                    for d in trackers:
                        if len(d) == 7:  # Проверяем, что трекер имеет все необходимые атрибуты
                            x1, y1, x2, y2, _, track_id, object_id = d
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                            cv2.putText(frame, f"Track ID: {track_id} Object ID: {object_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                yield frame

        finally:
            cap.release()

def generate_frames():
    detector = ObjectDetector()  # Создаем экземпляр класса ObjectDetector
    for frame in detector.video_detection():
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception as e:
        print(f"An error occurred: {e}")
