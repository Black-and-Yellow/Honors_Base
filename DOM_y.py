import math
from ultralytics import YOLO
import cv2
from sort import *
from RoadSurface_Extraction import extract_road_region

COCO_ClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"]


def intialize_y_direction():
    cap = cv2.VideoCapture("../Videos/footage_1.mp4")
    model = YOLO("./YOLOWeights/yolov8n.pt")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    tracking_obj = {}
    # total_foreground = extract_road_region("../Videos/footage_6.mp4")
    l_l_l_b = (-1, -1)
    l_l_r_b = (-1, -1)
    l_l_l_b_id = -1
    l_l_r_b_id = -1





    r_l_l_b = (-1, -1)
    r_l_r_b = (-1, -1)
    r_l_l_b_id = -1
    r_l_r_b_id = -1

    frame_count = 0

    while frame_count < 300:
        success, img = cap.read()
        frame_count += 1
        if not success:
            break

        height = img.shape[0]
        width = img.shape[1]
        # img = cv2.bitwise_and(img,img,mask=total_foreground)
        results = model(img, stream=True)
        detections = np.empty((0, 5))

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls[0]
                x, y, x_w, y_h = box.xyxy[0]
                x1, y1, x2, y2 = int(x), int(y), int(x_w), int(y_h)
                conf_val = math.ceil((box.conf[0] * 100)) / 100
                # cv2.putText(img, f'{cls_name}', (x1+10, y1 - 20), cv2.FONT_ITALIC, 0.7, (0, 255, 255), 2)
                current_arr = np.array([x1, y1, x2, y2, conf_val])
                detections = np.vstack((detections, current_arr))

        results_tracker = tracker.update(detections)

        for result in results_tracker:
            x, y, x_w, y_h, tracking_id = result
            x1, y1, x2, y2, tracking_id = int(x), int(y), int(x_w), int(y_h), int(tracking_id)
            # x1, y1, x2, y2, tracking_id = x, y, x_w, y_h, int(tracking_id)
            cv2.putText(img, f'Id-', (x1 + 10, y1 - 20), cv2.FONT_ITALIC, 0.55, (255, 255, 0), 2)
            cv2.putText(img, f'{tracking_id}', (x1 + 40, y1 - 20), cv2.FONT_ITALIC, 0.6, (0, 255, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            if tracking_id in tracking_obj:
                prev_pt = tracking_obj[tracking_id]
                curr_pt = (x1, y1, w, h)
                if curr_pt[1] >= prev_pt[1]+2:
                    if l_l_l_b == (-1, -1):
                        l_l_l_b = (x1, cy)
                    if l_l_r_b == (-1, -1):
                        l_l_r_b = (x1 + w, cy)

                    if x1 <= l_l_l_b[0]:
                        l_l_l_b = (x1, cy)
                        l_l_l_b_id = tracking_id
                    if x1 + w >= l_l_r_b[0]:
                        l_l_r_b = (x1 + w, cy)
                        l_l_r_b_id = tracking_id

                elif curr_pt[1] + 2 < prev_pt[1]:
                    if r_l_l_b == (-1, -1):
                        r_l_l_b = (x1, cy)
                    if r_l_r_b == (-1, -1):
                        r_l_r_b = (x1 + w, cy)

                    # cv2.circle(img, r_l_r_b, 2, (255, 255, 0), 2)
                    if x1 <= r_l_l_b[0]:
                        r_l_l_b = (x1, cy)
                        r_l_l_b_id = tracking_id
                    if x1 + w >= r_l_r_b[0]:
                        r_l_r_b = (x1 + w, cy)
                        r_l_r_b_id = tracking_id

            tracking_obj[tracking_id] = (x1, y1, w, h)

        # cv2.imshow("Video", img)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    return l_l_l_b_id, l_l_r_b_id, r_l_l_b_id, r_l_r_b_id



