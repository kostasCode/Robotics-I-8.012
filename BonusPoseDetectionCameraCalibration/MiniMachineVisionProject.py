import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
import string
from cv2_enumerate_cameras import enumerate_cameras

captured_hand_poses = []  # list of (landmarks, label)
label_counter = 0  # Global label counter

# ---------------------- Camera Detection ----------------------

def detect_cameras():
    available_cameras = []
    camera_backends = {}
    camera_names = {}

    for cam_info in enumerate_cameras(cv2.CAP_MSMF):
        index = cam_info.index
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            camera_backends[index] = cam_info.backend
            camera_names[index] = cam_info.name
            cap.release()

    return available_cameras, camera_backends, camera_names

# ---------------------- GUI Camera Selector ----------------------

def select_camera():
    def confirm_selection():
        global selected_camera
        selected = camera_var.get()
        if selected and int(selected) in available_cameras:
            selected_camera = int(selected)
            window.quit()
        else:
            messagebox.showerror("Error", "Invalid camera selection.")

    global available_cameras, camera_backends, camera_names
    available_cameras, camera_backends, camera_names = detect_cameras()

    window = tk.Tk()
    window.title("Select Camera Source")
    window.geometry("640x480")
    window.configure(bg="#f0f0f0")

    tk.Label(window, text="Available Cameras:", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=10)

    camera_var = tk.StringVar(value=str(available_cameras[0]) if available_cameras else "")

    if not available_cameras:
        tk.Label(window, text="No cameras detected!", font=("Arial", 12), fg="red", bg="#f0f0f0").pack()
    else:
        for cam in available_cameras:
            cam_desc = f"{camera_names[cam]} ({camera_backends[cam]})"
            tk.Radiobutton(window, text=cam_desc, variable=camera_var, value=str(cam), font=("Arial", 10), bg="#f0f0f0").pack(anchor="w", padx=20)

    ttk.Button(window, text="Start", command=confirm_selection).pack(pady=20)
    window.mainloop()
    window.destroy()

# ---------------------- Utility Functions ------------------------

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def convert_depth(z, z_min=-0.4, z_max=0.2, real_min=150, real_max=1000):
    z = -np.clip(z, z_min, z_max)
    return ((z_max - z) / (z_max - z_min)) * (real_max - real_min) + real_min

def project_point_3d_to_2d(point_3d):
    fx, fy = 1000, 1000
    width, height = 640, 480
    cx, cy = width // 2, height // 2

    #camera_matrix = np.array(
        # [
        #  [462.54083894 ,0,317.05326609],
        #  [0,347.12777165 , 242.48354991],
        #  [0,0,1]
        # ]
    #, dtype=np.float32)
    #dist_coeffs = np.array([ 0.18480259 , -0.70968216 , -0.00105616 , -0.00532912 ,  0.69399806], dtype=np.float32) 

    camera_matrix = np.array(
        [[fx, 0, cx], 
        [0, fy, cy], 
        [0, 0, 1]]
    , dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)   

    points_3d = np.array([[point_3d]], dtype=np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return tuple(points_2d.ravel().astype(int))

def next_label(counter):
    return string.ascii_uppercase[counter % 26]

def get_next_label():
    global label_counter
    label = next_label(label_counter)
    label_counter += 1
    return label

# ---------------------- Drawing Functions ------------------------

def draw_hand_axes(frame, landmarks, scale=30):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    index_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

    origin = np.array([(index_mcp.x * w) - cx, (index_mcp.y * h) - cy, convert_depth(index_mcp.z)])
    index_tip_3d = np.array([(index_tip.x * w) - cx, (index_tip.y * h) - cy, convert_depth(index_tip.z)])
    thumb_tip_3d = np.array([(thumb_tip.x * w) - cx, (thumb_tip.y * h) - cy, convert_depth(thumb_tip.z)])

    x_dir = normalize_vector(index_tip_3d - origin)
    t_dir = normalize_vector(thumb_tip_3d - origin)
    z_dir = normalize_vector(np.cross(x_dir, t_dir))
    y_dir = normalize_vector(np.cross(z_dir, x_dir))

    x_axis = origin + x_dir * scale
    y_axis = origin + y_dir * scale
    z_axis = origin + z_dir * scale

    origin_2d = project_point_3d_to_2d(origin)
    x_axis_2d = project_point_3d_to_2d(x_axis)
    y_axis_2d = project_point_3d_to_2d(z_axis)
    z_axis_2d = project_point_3d_to_2d(y_axis)

    cv2.arrowedLine(frame, origin_2d, x_axis_2d, (0, 0, 255), 3)  # X 
    cv2.arrowedLine(frame, origin_2d, y_axis_2d, (0, 255, 0), 3)  # Y 
    cv2.arrowedLine(frame, origin_2d, z_axis_2d, (255, 0, 0), 3)  # Z 

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'X', x_axis_2d, font, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, 'Y', y_axis_2d, font, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'Z', z_axis_2d, font, 0.7, (255, 0, 0), 2)

def draw_label_at_origin(frame, landmarks, label):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    index_mcp = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    origin_3d = np.array([(index_mcp.x * w) - cx, (index_mcp.y * h) - cy, convert_depth(index_mcp.z)])
    origin_2d = project_point_3d_to_2d(origin_3d)

    cv2.putText(frame, label, origin_2d, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# ---------------------- Main Hand Tracking -----------------------

def track_hands():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(selected_camera)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {selected_camera}.")
        return

    current_label = get_next_label()  # Start from A

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    draw_hand_axes(frame, hand_landmarks)

            # Draw stored poses and their labels
            for landmarks, label in captured_hand_poses:
                draw_hand_axes(frame, landmarks)
                draw_label_at_origin(frame, landmarks, label)

            # Show current label on screen
            if current_label:
                cv2.putText(frame, f'Current Label: {current_label}', (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Mini machine vision project TH20049', frame)
            key = cv2.waitKey(1)

            if key == 32:  # Spacebar
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        captured_hand_poses.append((hand_landmarks, current_label))
                        current_label = get_next_label()
                        break  # Only label one hand per frame
            elif key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- Main -------------------------------------

def main():
    select_camera()
    track_hands()

if __name__ == "__main__":
    main()
