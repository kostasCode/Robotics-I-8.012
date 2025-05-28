import cv2
import numpy as np

def calibrate_camera(chessboard_size=(7, 6), square_size=30.0, max_frames=15, camera_id=1):
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Σφάλμα: Δεν άνοιξε η κάμερα.")
        return None, None, None, None

    print(f">>> Οδηγίες:")
    print(f" - Τοποθετήστε τη σκακιέρα μέσα στο πράσινο πλαίσιο.")
    print(f" - Το πρόγραμμα θα αποθηκεύει αυτόματα {max_frames} λήψεις.")

    saved_frames = 0

    while saved_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("Σφάλμα λήψης frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # === Draw green guidance rectangle ===
        h, w = frame.shape[:2]
        rect_width = w // 2
        rect_height = h // 2
        x1 = (w - rect_width) // 2
        y1 = (h - rect_height) // 2
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if ret_corners:
            # Refine corner locations
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                              criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            saved_frames += 1
            print(f" - Frame {saved_frames} αποθηκεύτηκε!")

            # Draw detected corners
            cv2.drawChessboardCorners(frame, chessboard_size, corners_subpix, True)

        cv2.putText(frame, f"Saved frames: {saved_frames}/{max_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow('Calibration View', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            print(">>> Τερματισμός από τον χρήστη.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Σφάλμα: Δεν υπάρχουν αρκετά αποθηκευμένα frames για calibration.")
        return None, None, None, None

    print(">>> Υπολογισμός παραμέτρων κάμερας...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== ΑΠΟΤΕΛΕΣΜΑΤΑ CALIBRATION ===")
    print("\nIntrinsic Camera Matrix (K):\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs.ravel())
    print("\nReprojection Error:", ret)

    np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("\n>>> Τα αποτελέσματα σώθηκαν στο 'camera_calibration.npz'.")

    return camera_matrix, dist_coeffs, rvecs, tvecs

if __name__ == "__main__":
    calibrate_camera()
