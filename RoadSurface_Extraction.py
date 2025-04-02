import cv2
import numpy as np

def extract_background(video_path, num_frames=250, components=5, var_threshold=120):
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)

    gmm = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=var_threshold, detectShadows=False)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        mask = gmm.apply(frame)
    cap.release()

    return gmm, height, width

def extract_road_region(video_path):
    trained_gmm, height, width = extract_background(video_path)
    total_foreground = np.zeros((height, width), dtype=np.uint8)
    frame_count = 0
    cap2 = cv2.VideoCapture(video_path)
    prev_frame = None

    while True:
        # print(frame_count)
        if frame_count > 400:
            print("Road Mask Formed...")
            break

        ret2, img = cap2.read()
        if not ret2:
            break
        frame_count += 1
        img_blurred = cv2.GaussianBlur(img, (7, 7), 0)
        # cv2.imshow("Blurred Img", img_blurred)

        # Applying frame differencing
        if prev_frame is not None:
            diff_frame = cv2.absdiff(prev_frame, img_blurred)
            # cv2.imshow("MainImage", diff_frame)
            diff_gray = cv2.cvtColor(diff_frame, cv2.COLOR_BGR2GRAY)
            _, diff_thresh = cv2.threshold(diff_gray, 40, 255, cv2.THRESH_BINARY)
            # cv2.imshow("Threshold Img", diff_thresh)
            foreground_mask = trained_gmm.apply(img_blurred)
            # cv2.imshow("foreground Img mask", foreground_mask)
            foreground_mask = cv2.bitwise_and(foreground_mask, diff_thresh)  # Combine with GMM mask

            # cv2.imshow("foreground Img mask", foreground_mask)
        else:
            foreground_mask = trained_gmm.apply(img_blurred)

        if prev_frame is not None:
            total_foreground = cv2.bitwise_or(total_foreground, foreground_mask)

        prev_frame = img_blurred.copy()

        foreground = cv2.bitwise_and(img, img, mask=foreground_mask)
        background = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(foreground_mask))

        # cv2.imshow("MainImage", img_blurred)
        # cv2.imshow('foreground.png', total_foreground)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap2.release()
    cv2.destroyAllWindows()

    return total_foreground


# video_path = '../Videos/footage_2.mp4'
# total_foreground = extract_road_region(video_path)

# Dilate the mask to fill gaps
# kernel = np.ones((5, 5), np.uint8)
# total_foreground = cv2.dilate(total_foreground, kernel, iterations=5)

# cap3 = cv2.VideoCapture(video_path)
#
# while True:
#     ret3, img = cap3.read()
#     if not ret3:
#         break
#     light_red = np.full_like(img, (255, 200, 200), dtype=np.uint8)
#     light_red_mask = cv2.bitwise_and(light_red, light_red, mask=total_foreground)
#     result = cv2.addWeighted(img, 1, light_red_mask, 0.5, 0)
#
#     frame_cropped = cv2.bitwise_and(img, img, mask=total_foreground)
#     cv2.imshow('Result', frame_cropped)
#     cv2.imshow('foreground.png', total_foreground)
#     cv2.imshow("Main Img", img)
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
# cap3.release()
# cv2.destroyAllWindows()

