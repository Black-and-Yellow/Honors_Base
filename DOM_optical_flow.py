import cv2
import numpy as np
from sort import Sort  # Ensure the SORT library is available
import csv
import sys

class VehicleTracker:
    def __init__(self, threshold=5):
        self.tracker = Sort()
        self.vehicle_dict = {}
        self.frame_boundaries = []
        self.road_boundaries = []
        self.boundaries_left = []
        self.boundaries_right = []
        self.global_min_left = None
        self.global_max_left = None
        self.global_min_right = None
        self.global_max_right = None
        self.threshold = threshold
        self.direction_count = {
            "left_to_right": 0,
            "right_to_left": 0,
            "top_to_bottom": 0,
            "bottom_to_top": 0
        }

    def update(self, detections):
        if detections:
            detections = np.array(detections)
            tracked_objects = self.tracker.update(detections)
            current_ids = set()
            for obj in tracked_objects:
                obj_id = int(obj[4])
                bbox = obj[:4]
                current_ids.add(obj_id)
                if obj_id in self.vehicle_dict:
                    self.update_direction(obj_id, self.vehicle_dict[obj_id]['bbox'], bbox)
                    self.vehicle_dict[obj_id]['bbox'] = bbox
                    self.vehicle_dict[obj_id]['path'].append(bbox)
                else:
                    self.vehicle_dict[obj_id] = {'id': obj_id, 'bbox': bbox, 'path': [bbox], 'direction': None}
            obsolete_ids = set(self.vehicle_dict.keys()) - current_ids
            for obj_id in obsolete_ids:
                del self.vehicle_dict[obj_id]

    def update_direction(self, obj_id, prev_bbox, new_bbox):
        if prev_bbox is None:
            return
        
        x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
        x1_new, y1_new, x2_new, y2_new = new_bbox

        x_diff = x1_new - x1_prev
        y_diff = y1_new - y1_prev

        new_direction = None

        if abs(x_diff) > abs(y_diff):
            if x_diff > self.threshold:
                new_direction = "left_to_right"
            elif x_diff < -self.threshold:
                new_direction = "right_to_left"
        else:
            if y_diff > self.threshold:
                new_direction = "top_to_bottom"
            elif y_diff < -self.threshold:
                new_direction = "bottom_to_top"

        if new_direction is not None:
            if self.vehicle_dict[obj_id]['direction'] != new_direction:
                if self.vehicle_dict[obj_id]['direction']:
                    self.direction_count[self.vehicle_dict[obj_id]['direction']] -= 1
                self.direction_count[new_direction] += 1
                self.vehicle_dict[obj_id]['direction'] = new_direction

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
            direction = vehicle.get('direction', 'Unknown')
            cv2.putText(frame, f"Direction: {direction}",
                        (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
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

def find_vehicle_boundaries(video_path, max_frames=250):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return [], [], {}

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    tracker = VehicleTracker()
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]
        roi_frame = frame[int(height * 0.4):, :]

        fg_mask = bg_subtractor.apply(roi_frame)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        fg_mask = cv2.threshold(fg_mask, 190, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x += 0
            y += int(height * 0.4)
            if w > 30 and h > 30:
                detections.append([x, y, x + w, y + h, 1])

        tracker.update(detections)
        tracker.record_boundaries(frame_count)

    boundaries = tracker.frame_boundaries
    road_boundaries = tracker.road_boundaries
    direction_count = tracker.direction_count

    cap.release()

    return boundaries, road_boundaries, direction_count

def save_boundaries_to_csv(boundaries, road_boundaries, direction_count, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        for boundary in boundaries:
            writer.writerow(boundary)
        
        writer.writerow([])
        writer.writerow(["Vehicle_ID", "Initial_BBox", "Current_BBox"])
        for road_boundary in road_boundaries:
            writer.writerow(road_boundary)
        
        writer.writerow([])
        writer.writerow(["Direction", "Count"])
        for direction, count in direction_count.items():
            writer.writerow([direction, count])

def mark_vehicle_boundaries(video_path, output_path, output_csv, max_frames=250):
    boundaries, road_boundaries, direction_count = find_vehicle_boundaries(video_path, max_frames)
    if not boundaries and not road_boundaries:
        print("Error: No boundaries found. Ensure the video path is correct.")
        return

    save_boundaries_to_csv(boundaries, road_boundaries, direction_count, output_csv)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
    tracker = VehicleTracker()
    frame_count = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width = frame.shape[:2]
        roi_frame = frame[int(height * 0.4):, :]

        fg_mask = bg_subtractor.apply(roi_frame)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        fg_mask = cv2.threshold(fg_mask, 190, 255, cv2.THRESH_BINARY)[1]
        fg_mask = cv2.erode(fg_mask, None, iterations=2)
        fg_mask = cv2.dilate(fg_mask, None, iterations=3)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x += 0
            y += int(height * 0.4)
            if w > 30 and h > 30:
                detections.append([x, y, x + w, y + h, 1])

        tracker.update(detections)
        tracker.draw(frame)
        tracker.detect_direction_and_draw_boundaries(frame)
        tracker.draw_static_boundaries(frame)

        out.write(frame)

    cap.release()
    out.release()

    print("Final output video saved as:", output_path)
    print("CSV file with boundaries and directions saved as:", output_csv)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python DOM_optical_flow.py <input_video> <output_video> <output_csv> <max_frames>")
    else:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        output_csv = sys.argv[3]
        max_frames = int(sys.argv[4])
        mark_vehicle_boundaries(video_path, output_path, output_csv, max_frames)
