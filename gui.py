import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import subprocess
from PIL import Image, ImageTk
import re
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to extract frame from video
def extract_frame_from_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        raise ValueError("Could not extract frame from video.")

# Function to merge lines based on proximity
def merge_lines(lines, threshold=150):
    if lines is None:
        return []
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        merged = False
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            if (abs(mx1 - x1) < threshold and abs(my1 - y1) < threshold and
                abs(mx2 - x2) < threshold and abs(my2 - y2) < threshold):
                merged = True
                break
        if not merged:
            merged_lines.append([x1, y1, x2, y2])
    return merged_lines

# Function to count unique lanes
def count_lanes(merged_lines):
    if len(merged_lines) == 0:
        return 0
    slopes = []
    for line in merged_lines:
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
        slopes.append(slope)
    unique_slopes = np.unique(np.array(slopes))
    unique_slopes = unique_slopes[unique_slopes != np.inf]
    num_lanes = len(unique_slopes)
    return num_lanes

# Main function to process the frame
def process_frame(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 150, 100)

    height, width = edges.shape
    roi = np.array([[(0, height), (width, height), (width, int(height * 0.5)), (0, int(height * 0.5))]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=70, maxLineGap=7)
    merged_lines = merge_lines(lines)
    num_lanes = count_lanes(merged_lines)

    line_image = np.copy(image)
    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)

    return line_image, num_lanes

class VideoProcessor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Processing GUI")
        self.geometry("1000x700")

        self.video_path = ""
        self.output_path = "processed_output.avi"
        self.output_csv = "boundaries.csv"
        self.csv_folder_path = ""

        self.create_widgets()

    def create_widgets(self):
        # Row 1: Display Video
        self.upload_btn = tk.Button(self, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(pady=10)

        self.video_label = tk.Label(self)
        self.video_label.pack()

        # Row 2: Enter Number of Rows and Columns
        row_col_frame = tk.Frame(self)
        row_col_frame.pack(pady=10)

        self.row_label = tk.Label(row_col_frame, text="Enter Number of Rows:")
        self.row_label.grid(row=0, column=0, padx=5, pady=5)
        self.row_entry = tk.Entry(row_col_frame)
        self.row_entry.grid(row=0, column=1, padx=5, pady=5)

        self.col_label = tk.Label(row_col_frame, text="Enter Number of Columns:")
        self.col_label.grid(row=0, column=2, padx=5, pady=5)
        self.col_entry = tk.Entry(row_col_frame)
        self.col_entry.grid(row=0, column=3, padx=5, pady=5)

        # Row 3: Ask User Choice and Number of Frames
        user_choice_frame = tk.Frame(self)
        user_choice_frame.pack(pady=10)

        self.user_choice_label = tk.Label(user_choice_frame, text="User Choice:")
        self.user_choice_label.grid(row=0, column=0, padx=5, pady=5)
        self.user_choice_entry = tk.Entry(user_choice_frame)
        self.user_choice_entry.grid(row=0, column=1, padx=5, pady=5)

        self.frame_label = tk.Label(user_choice_frame, text="Number of Frames:")
        self.frame_label.grid(row=0, column=2, padx=5, pady=5)
        self.frame_entry = tk.Entry(user_choice_frame)
        self.frame_entry.grid(row=0, column=3, padx=5, pady=5)

        # Row 4: Display Buttons
        self.btn_frame = tk.Frame(self)
        self.btn_frame.pack(pady=10)

        self.blob_btn = tk.Button(self.btn_frame, text="Blob Tracking", command=lambda: self.process_video('blobtracking1.py'))
        self.blob_btn.grid(row=0, column=0, padx=5, pady=5)

        self.optical_btn = tk.Button(self.btn_frame, text="Optical Flow", command=lambda: self.process_video('DOM_optical_flow.py'))
        self.optical_btn.grid(row=0, column=1, padx=5, pady=5)

        self.yolo_btn = tk.Button(self.btn_frame, text="YOLO Sort", command=lambda: self.process_video('yolo.py'))
        self.yolo_btn.grid(row=0, column=2, padx=5, pady=5)

        # Row 5: Lane Detection Button
        self.lane_btn = tk.Button(self.btn_frame, text="Find Number of Lanes", command=self.find_lanes)
        self.lane_btn.grid(row=1, column=0, padx=5, pady=5)
        
        
        self.lane_btn = tk.Button(self.btn_frame, text="direction",command=lambda: self.process_video('direction.py'))
        self.lane_btn.grid(row=1, column=1, padx=5, pady=5)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            self.video_path = file_path
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.imgtk = imgtk
            cap.release()

    def ask_save_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.csv_folder_path = folder_path

    def process_video(self, script_name):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first.")
            return

        try:
            num_frames = int(self.frame_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of frames.")
            return

        try:
            rows = int(self.row_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of rows.")
            return

        try:
            cols = int(self.col_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of columns.")
            return

        user_choice = self.user_choice_entry.get()
        if not user_choice:
            messagebox.showerror("Error", "Please enter a valid user choice.")
            return

        self.ask_save_folder()
        if not self.csv_folder_path:
            messagebox.showerror("Error", "Please select a folder to save the CSV file.")
            return

        self.output_csv = os.path.join(self.csv_folder_path, "boundaries_opticalflow.csv")

        try:
            result = subprocess.run(['python', script_name, self.video_path, self.output_path, self.output_csv, str(num_frames)], capture_output=True, text=True)
            print("Result stdout:", result.stdout)  # Print stdout to debug
            print("Result stderr:", result.stderr)  # Print stderr to debug
            if result.returncode == 0:
                output_video_path_match = re.search(r'Final output video saved as: (.+\.avi)', result.stdout)
                if output_video_path_match:
                    output_video_path = output_video_path_match.group(1)
                    self.play_video(output_video_path)
                else:
                    messagebox.showerror("Error", "Output video path not found in subprocess output.")
            else:
                messagebox.showerror("Error", f"An error occurred: {result.stderr}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def save_csv(self):
        if not self.csv_folder_path:
            messagebox.showerror("Error", "No folder selected to save the CSV file.")
            return

        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialdir=self.csv_folder_path)
        if csv_path:
            try:
                with open(self.output_csv, 'r') as src, open(csv_path, 'w') as dst:
                    dst.write(src.read())
                messagebox.showinfo("Success", f"Boundaries CSV saved to {csv_path}")
            except FileNotFoundError:
                messagebox.showerror("Error", "boundaries.csv not found. Ensure the video has been processed.")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the CSV: {e}")

    def play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.imgtk = imgtk
                self.video_label.after(33, update_frame)  # 30 frames per second ~ 33ms per frame
            else:
                cap.release()

        update_frame()

    def find_lanes(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first.")
            return

        try:
            frame_number = int(self.frame_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid frame number.")
            return

        try:
            frame = extract_frame_from_video(self.video_path, frame_number)
            processed_frame, num_lanes = process_frame(frame)
            print("Number of lanes detected:", num_lanes)

            # Display results
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame_rgb)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.imgtk = imgtk

            messagebox.showinfo("Lanes Detected", f"Number of lanes detected: {num_lanes}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    app = VideoProcessor()
    app.mainloop()
