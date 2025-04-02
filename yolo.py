import cv2
import numpy as np
from sort2 import Sort  # Assuming you have your custom SORT implementation in sort2
import csv
import sys
from DOM_y import intialize_y_direction
import math
from ultralytics import YOLO


class VehicleTracker:
    def __init__(self, yolo_model_path="./YOLOWeights/yolov8n.pt", max_age=20, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.yolo_model = YOLO(yolo_model_path)
        self.vehicle_dict = {}
        self.frame_boundaries = []
        self.road_boundaries = []
        self.boundaries_left = []
        self.boundaries_right = []
        self.global_min_left = None
        self.global_max_left = None
        self.global_min_right = None
        self.global_max_right = None

    def update_from_yolo_sort(self, frame):
        detections = self.detect_and_process(frame)
        self.update(detections)

    def detect_and_process(self, frame):
        results = self.yolo_model(frame, stream=True)
        detections = np.empty((0, 5))

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf_val = math.ceil((box.conf[0] * 100)) / 100
                current_arr = np.array([x1, y1, x2, y2, conf_val])
                detections = np.vstack((detections, current_arr))

        return detections

    def update(self, detections):
        if detections.size > 0:
            tracked_objects = self.tracker.update(detections)
            current_ids = set()
            for obj in tracked_objects:
                obj_id = int(obj[4])
                bbox = obj[:4]
                current_ids.add(obj_id)
                if obj_id in self.vehicle_dict:
                    self.vehicle_dict[obj_id]['bbox'] = bbox
                    self.vehicle_dict[obj_id]['path'].append(bbox)
                else:
                    self.vehicle_dict[obj_id] = {'id': obj_id, 'bbox': bbox, 'path': [bbox]}
            obsolete_ids = set(self.vehicle_dict.keys()) - current_ids
            for obj_id in obsolete_ids:
                del self.vehicle_dict[obj_id]

    def get_min_max_coordinates(self):
        if not self.vehicle_dict:
            return self.global_min_left, self.global_max_left, self.global_min_right, self.global_max_right

        min_left = float('inf')
        max_left = float('-inf')
        min_right = float('inf')
        max_right = float('-inf')

        for vehicle in self.vehicle_dict.values():
            x1, y1, x2, y2 = vehicle['bbox']
            min_left = min(min_left, x1)
            max_left = max(max_left, x2)
            min_right = min(min_right, y1)
            max_right = max(max_right, y2)

        if self.global_min_left is None or min_left < self.global_min_left:
            self.global_min_left = min_left
        if self.global_max_left is None or max_left > self.global_max_left:
            self.global_max_left = max_left
        if self.global_min_right is None or min_right < self.global_min_right:
            self.global_min_right = min_right
        if self.global_max_right is None or max_right > self.global_max_right:
            self.global_max_right = max_right

        return self.global_min_left, self.global_max_left, self.global_min_right, self.global_max_right

    def record_boundaries(self, frame_id):
        min_left, max_left, min_right, max_right = self.get_min_max_coordinates()
        self.frame_boundaries.append([frame_id, min_left, max_left, min_right, max_right])
        print(f"Frame {frame_id}: Min_Left = {min_left}, Max_Left = {max_left}, Min_Right = {min_right}, Max_Right = {max_right}")

    def draw(self, frame):
        for vehicle in self.vehicle_dict.values():
            x1, y1, x2, y2 = vehicle['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {vehicle['id']}, Coordinates: ({x1},{y1}) - ({x2},{y2})",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            path = vehicle['path']
            if len(path) > 1:
                for i in range(1, len(path)):
                    start_point1 = (int(path[i-1][0]), int((path[i-1][1] + path[i-1][3]) / 2))
                    end_point1 = (int(path[i][0]), int((path[i][1] + path[i][3]) / 2))
                    start_point2 = (int(path[i-1][2]), int((path[i-1][1] + path[i-1][3]) / 2))
                    end_point2 = (int(path[i][2]), int((path[i][1] + path[i][3]) / 2))
                    cv2.line(frame, start_point1, end_point1, (0, 0, 255), 2)
                    cv2.line(frame, start_point2, end_point2, (0, 0, 255), 2)
                    self.boundaries_left.append((start_point1, end_point1))
                    self.boundaries_right.append((start_point2, end_point2))

    def detect_direction_and_draw_boundaries(self, frame):
        for vehicle in self.vehicle_dict.values():
            x1, y1, x2, y2 = vehicle['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            self.road_boundaries.append((vehicle['id'], x1, y1, x2, y2))

    def draw_static_boundaries(self, frame):
        for boundary in self.boundaries_left:
            cv2.line(frame, boundary[0], boundary[1], (255, 0, 0), 2)
        for boundary in self.boundaries_right:
            cv2.line(frame, boundary[0], boundary[1], (0, 0, 255), 2)

def mark_vehicle_boundaries(video_path, output_path, output_csv, max_frames=250):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    tracker = VehicleTracker()
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        tracker.update_from_yolo_sort(frame)
        tracker.record_boundaries(frame_count)
        tracker.draw(frame)
        tracker.detect_direction_and_draw_boundaries(frame)
        tracker.draw_static_boundaries(frame)

        out.write(frame)

    cap.release()
    out.release()

    print("Final output video saved as:", output_path)
    print("Boundaries recorded in CSV file:", output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python blobtracking1.py <input_video> <output_video> <output_csv> <max_frames>")
    else:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        output_csv = sys.argv[3]
        max_frames = int(sys.argv[4])
        mark_vehicle_boundaries(video_path, output_path, output_csv, max_frames)
