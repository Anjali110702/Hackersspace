import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os

class VirtualTryOnApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Virtual Try-On")
        self.window.geometry("800x600")

        # Initialize MediaPipe Pose solution
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Load an example watch image (transparent PNG)
        watch_img_path = "C:/Users/anjal/Downloads/watch.png"
        if not os.path.exists(watch_img_path):
            print("Error: Watch image not found.")
            self.window.destroy()
            return

        self.watch_img = cv2.imread(watch_img_path, cv2.IMREAD_UNCHANGED)
        if self.watch_img is None:
            print("Error: Unable to load watch image.")
            self.window.destroy()
            return

        # Check if the image has an alpha channel
        self.has_alpha = self.watch_img.shape[2] == 4

        # Create a label for displaying the video feed
        self.video_label = tk.Label(window)
        self.video_label.pack()

        # Create a button to start/stop the virtual try-on
        self.start_button = tk.Button(window, text="Start Virtual Try-On", command=self.toggle_try_on)
        self.start_button.pack()

        self.cap = None
        self.is_running = False

    def toggle_try_on(self):
        if self.is_running:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.start_button.config(text="Start Virtual Try-On")
        else:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Webcam not found or unable to access.")
                self.window.destroy()
                return
            self.start_button.config(text="Stop Virtual Try-On")
            self.update_frame()

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Unable to capture video frame.")
            self.stop_try_on()
            return

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for wrists
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            left_wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
            right_wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))

            # Calculate the distance between wrists
            wrist_distance = np.linalg.norm(np.array(left_wrist_coords) - np.array(right_wrist_coords))

            if wrist_distance > 0:  # Ensure the distance is positive
                # Resize the watch image to fit on the wrist
                watch_height = int(self.watch_img.shape[0] * (wrist_distance / self.watch_img.shape[1]))
                resized_watch = cv2.resize(self.watch_img, (int(wrist_distance), watch_height))

                # Position the watch on the left wrist
                x_offset = left_wrist_coords[0] - int(resized_watch.shape[1] / 2)
                y_offset = left_wrist_coords[1] - int(resized_watch.shape[0] / 2)
                y1, y2 = y_offset, y_offset + resized_watch.shape[0]
                x1, x2 = x_offset, x_offset + resized_watch.shape[1]

                # Ensure the coordinates are within frame boundaries
                if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                    if self.has_alpha:
                        alpha_watch = resized_watch[:, :, 3] / 255.0
                        alpha_frame = 1.0 - alpha_watch

                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_watch * resized_watch[:, :, c] +
                                                      alpha_frame * frame[y1:y2, x1:x2, c])
                    else:
                        # No alpha channel, just overlay the image
                        frame[y1:y2, x1:x2] = resized_watch[:, :, :3]

        # Convert the frame to Image format and display it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)

        self.video_label.config(image=photo)
        self.video_label.image = photo

        self.window.after(10, self.update_frame)

    def stop_try_on(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(text="Start Virtual Try-On")

if __name__ == "__main__":
    root = tk.Tk()
    app = VirtualTryOnApp(root)
    root.mainloop()
