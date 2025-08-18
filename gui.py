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

# MERGED GUI CLASS

class VideoProcessor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video Processing GUI")
        self.geometry("830x700")
        self.configure(bg="#f0f0f0")

        # Variables from both scripts
        self.video_path = ""
        self.output_path = "processed_output.avi"
        self.output_csv = "boundaries.csv"
        self.csv_folder_path = ""

        self.create_widgets()

    def _on_canvas_configure(self, event):
        """Resizes the scrollable_frame to match the canvas width."""
        canvas_width = event.width
        self.canvas.itemconfig(self.frame_window, width=canvas_width)

    def create_widgets(self):
        # --- Base GUI Structure ---
        self.canvas = tk.Canvas(self, bg="#f0f0f0", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f0f0f0")

        # This binding updates the scrollregion when the size of the scrollable_frame content changes
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Place the scrollable_frame inside the canvas and store its ID
        self.frame_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="n")

        # **NEW**: Bind the canvas resizing event to our new function
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # --- Styling ---
        frame_style = {"padx": 10, "pady": 10, "bd": 2, "relief": tk.RIDGE, "bg": "#ffffff"}
        label_style = {"padx": 5, "pady": 5, "bg": "#ffffff", "font": ("Arial", 12)} # Changed bg to match frame
        entry_style = {"font": ("Arial", 12)}
        button_style = {"font": ("Arial", 12), "cursor": "hand2"}

        # --- Video Upload and Display ---
        self.upload_btn = tk.Button(self.scrollable_frame, text="Upload Video", command=self.upload_video, **button_style)
        self.upload_btn.pack(pady=20)

        self.video_label = tk.Label(self.scrollable_frame, bg="#000")
        self.video_label.pack()

        # --- Frame for All User Inputs ---
        input_frame = tk.Frame(self.scrollable_frame, **frame_style)
        input_frame.pack(pady=10)

        self.row_label = tk.Label(input_frame, text="Enter Number of Rows:", **label_style)
        self.row_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.row_entry = tk.Entry(input_frame, **entry_style)
        self.row_entry.grid(row=0, column=1, padx=5, pady=5)

        self.col_label = tk.Label(input_frame, text="Enter Number of Columns:", **label_style)
        self.col_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')
        self.col_entry = tk.Entry(input_frame, **entry_style)
        self.col_entry.grid(row=0, column=3, padx=5, pady=5)

        self.user_choice_label = tk.Label(input_frame, text="User Choice:", **label_style)
        self.user_choice_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.user_choice_entry = tk.Entry(input_frame, **entry_style)
        self.user_choice_entry.grid(row=1, column=1, padx=5, pady=5)

        self.frame_label = tk.Label(input_frame, text="Number of Frames:", **label_style)
        self.frame_label.grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.frame_entry = tk.Entry(input_frame, **entry_style)
        self.frame_entry.grid(row=1, column=3, padx=5, pady=5)

        # --- Frame for All Buttons ---
        self.btn_frame = tk.Frame(self.scrollable_frame, **frame_style)
        self.btn_frame.pack(pady=10)

        self.blob_btn = tk.Button(self.btn_frame, text="Blob Tracking", command=lambda: self.process_video('blobtracking1.py'), **button_style)
        self.blob_btn.grid(row=0, column=0, padx=5, pady=5)

        self.optical_btn = tk.Button(self.btn_frame, text="Optical Flow", command=lambda: self.process_video('DOM_optical_flow.py'), **button_style)
        self.optical_btn.grid(row=0, column=1, padx=5, pady=5)

        self.yolo_btn = tk.Button(self.btn_frame, text="YOLO Sort", command=lambda: self.process_video('yolo.py'), **button_style)
        self.yolo_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.lane_btn = tk.Button(self.btn_frame, text="Find Number of Lanes", command=self.find_lanes, **button_style)
        self.lane_btn.grid(row=1, column=0, padx=5, pady=5)
        
        self.direction_btn = tk.Button(self.btn_frame, text="Direction",command=lambda: self.process_video('direction.py'), **button_style)
        self.direction_btn.grid(row=1, column=1, padx=5, pady=5)

        self.seq_btn = tk.Button(self.btn_frame, text="Sequential Processing", command=lambda: self.run_processing_script(self.video_path, self.row_entry.get(), self.col_entry.get(), self.user_choice_entry.get(), "Sequential"), **button_style)
        self.seq_btn.grid(row=2, column=0, padx=5, pady=10)

        self.parallel_btn = tk.Button(self.btn_frame, text="Parallel Processing", command=lambda: self.run_processing_script(self.video_path, self.row_entry.get(), self.col_entry.get(), self.user_choice_entry.get(), "Parallel"), **button_style)
        self.parallel_btn.grid(row=2, column=1, padx=5, pady=10, columnspan=2)


    # --- Methods from first and second scripts ---

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
    
    # --- ALL BACKEND METHODS FROM YOUR FIRST SCRIPT (UNCHANGED) ---
    
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

        # These inputs are not used by the original script's logic for these buttons, but kept for consistency
        try:
            rows = int(self.row_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of rows.")
        try:
            cols = int(self.col_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of columns.")

        user_choice = self.user_choice_entry.get()
        if not user_choice:
            messagebox.showerror("Error", "Please enter a valid user choice.")

        self.ask_save_folder()
        if not self.csv_folder_path:
            messagebox.showerror("Error", "Please select a folder to save the CSV file.")
            return

        self.output_csv = os.path.join(self.csv_folder_path, "boundaries_opticalflow.csv")
        
        # ***** CHANGED LINE HERE *****
        full_script_path = os.path.join('IIT-Hyd-Intern', script_name)

        try:
            result = subprocess.run(['python', full_script_path, self.video_path, self.output_path, self.output_csv, str(num_frames)], capture_output=True, text=True)
            print("Result stdout:", result.stdout)
            print("Result stderr:", result.stderr)
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

    def play_video(self, video_path):
        """A consolidated method to play any video file."""
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
                self.video_label.after(33, update_frame)
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

            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(processed_frame_rgb)
            img = img.resize((640, 480), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.imgtk = imgtk

            messagebox.showinfo("Lanes Detected", f"Number of lanes detected: {num_lanes}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def run_processing_script(self, video_path, rows, cols, user_choice, seq_par_choice):
        # This path was already correct in the original script
        script_name = 'IIT-Hyd-Intern/seq.py' if seq_par_choice == "Sequential" else 'IIT-Hyd-Intern/rparallel_he3.py'

        try:
            result = subprocess.run(['python', script_name, video_path, str(rows), str(cols), user_choice, self.output_path], capture_output=True, text=True)
            print("Result stdout:", result.stdout)
            print("Result stderr:", result.stderr)
            if result.returncode == 0:
                self.display_processed_video()
                messagebox.showinfo("Success", "Video Processed")
            else:
                messagebox.showerror("Error", f"An error occurred: {result.stderr}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            
    def display_processed_video(self):
        self.play_video(self.output_path)


if __name__ == "__main__":
    app = VideoProcessor()
    app.mainloop()