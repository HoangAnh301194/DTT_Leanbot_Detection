"""
=============================================================
BƯỚC 2: CALIBRATE CAMERA TỪ ẢNH ĐÃ CHỤP
=============================================================
Nội dung :
1. Đảm bảo đã chụp đủ 15 trở lên số ảnh bằng capture_calibration_images.py
2. Chạy script này để tính toán ma trận thấu kính
3. Kết quả tham số sẽ được lưu vào calibration_result.npz 
4. Hệ thống sẽ tự tạo thư mục comparison_results và lưu lại cặp ảnh gốc/khử méo để so sánh

Kết quả bao gồm:
- Camera Matrix (K): ma trận nội tham số
- Distortion Coefficients: hệ số méo ống kính
- Rotation vectors (rvecs): vector quay cho mỗi ảnh
- Translation vectors (tvecs): vector tịnh tiến cho mỗi ảnh
- Re-projection Error: sai số chiếu lại (càng nhỏ càng tốt, tối đa <= 0.5)
"""

import cv2
import numpy as np
import os
import glob

# ======================== CẤU HÌNH ========================
CHECKERBOARD = (4, 4)     # Số inner corners (phải giống file capture)
SQUARE_SIZE = 30.0        # Kích thước mỗi ô vuông (mm). ĐO LẠI bằng thước!
IMAGE_DIR = "./calibration_images"   # Thư mục chứa ảnh đã chụp
OUTPUT_FILE = "calibration_result.npz"  # File lưu kết quả
CAMERA_INDEX = 1          # Index camera để demo realtime
CAMERA_WIDTH = 1920       # Độ phân giải ngang (Phải khớp với lúc chụp ảnh)
CAMERA_HEIGHT = 1080      # Độ phân giải dọc (Phải khớp với lúc chụp ảnh)
# ===========================================================


def main():
    print("=" * 60)
    print("  CAMERA CALIBRATION - Hikvision DS-U04")
    print("=" * 60)

    # -------- BƯỚC 1: Tạo tọa độ 3D thực (Object Points) --------
    # Mỗi corner có tọa độ (X*SQUARE_SIZE, Y*SQUARE_SIZE, 0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale theo kích thước ô thực tế (mm)

    # Danh sách lưu points cho tất cả ảnh
    objpoints = []  # 3D points (giống nhau cho mọi ảnh)
    imgpoints = []  # 2D points (khác nhau tùy ảnh)

    # Tiêu chí dừng cho cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # -------- BƯỚC 2: Đọc và xử lý từng ảnh --------
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

    if len(images) == 0:
        print(f"[ERROR] Không tìm thấy ảnh trong {IMAGE_DIR}")
        print(" -> Chạy capture_calibration_images.py trước!")
        return

    print(f"\n[INFO] Tìm thấy {len(images)} ảnh trong {IMAGE_DIR}")
    print("[INFO] Đang xử lý...\n")

    img_shape = None
    success_count = 0

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # (width, height)

        # Tìm corners
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            success_count += 1
            objpoints.append(objp)

            # Tinh chỉnh corners với độ chính xác sub-pixel
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Hiển thị corners (tùy chọn)
            display = img.copy()
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners2, ret)

            # Resize để hiển thị
            h, w = display.shape[:2]
            if w > 960:
                scale = 960 / w
                display = cv2.resize(display, (int(w * scale), int(h * scale)))

            cv2.imshow("Corners Detected", display)
            cv2.waitKey(300)  # Hiện 300ms mỗi ảnh

            basename = os.path.basename(fname)
            print(f" [Done] {basename} — Corners detected ({success_count})")
        else:
            basename = os.path.basename(fname)
            print(f" [Error] {basename} — Không tìm thấy corners (bỏ qua)")

    cv2.destroyAllWindows()

    if success_count < 10:
        print(f"\n[ERROR] Chỉ detect được {success_count} ảnh. Nên có ít nhất 10!")
        if success_count < 3:
            print("[ERROR] Không đủ ảnh để calibrate. Chụp lại!")
            return

    # -------- BƯỚC 3: CALIBRATE --------
    print(f"\n[INFO] Calibrating với {success_count} ảnh...")
    print("[INFO] Đang tính toán... (có thể mất vài giây)\n")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # -------- BƯỚC 4: HIỂN THỊ KẾT QUẢ --------
    print("=" * 60)
    print("  KẾT QUẢ CALIBRATION")
    print("=" * 60)

    print(f"\n Re-projection Error: {ret:.4f} pixel")
    if ret < 0.5:
        print(f" Calibration thành công với sai số {ret:.4f} pixel.")
    elif ret < 1.0:
        print(f" Calibartion tương đối ở mức {ret:.4f} pixel.Cần chụp thêm các góc ảnh khác để tính toán lại")
    else:
        print(f" Calibration không thành công với sai số {ret:.4f} pixel.Cần chụp lại ảnh cẩn thận hơn.")

    print(f"\n Camera Matrix (K):")
    print(camera_matrix)
    print(f"\n  fx = {camera_matrix[0, 0]:.2f} pixel")
    print(f"  fy = {camera_matrix[1, 1]:.2f} pixel")
    print(f"  cx = {camera_matrix[0, 2]:.2f} pixel (tâm quang học X)")
    print(f"  cy = {camera_matrix[1, 2]:.2f} pixel (tâm quang học Y)")

    print(f"\n Distortion Coefficients:")
    print(dist_coeffs)
    print(f"\n  k1 = {dist_coeffs[0, 0]:.6f}  (radial)")
    print(f"  k2 = {dist_coeffs[0, 1]:.6f}  (radial)")
    print(f"  p1 = {dist_coeffs[0, 2]:.6f}  (tangential)")
    print(f"  p2 = {dist_coeffs[0, 3]:.6f}  (tangential)")
    print(f"  k3 = {dist_coeffs[0, 4]:.6f}  (radial)")

    # -------- BƯỚC 5: LƯU KẾT QUẢ --------
    np.savez(OUTPUT_FILE,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=np.array(rvecs, dtype=object),
             tvecs=np.array(tvecs, dtype=object),
             reprojection_error=ret,
             image_size=img_shape,
             checkerboard_size=CHECKERBOARD,
             square_size=SQUARE_SIZE)

    print(f"\n Đã lưu kết quả vào: {OUTPUT_FILE}")

    # -------- BƯỚC 6: LIVE DEMO UNDISTORT --------
    print(f"\n[INFO] Bắt đầu Live Demo khử méo (undistort) từ Camera {CAMERA_INDEX}...")
    print("  Nháy vào cửa sổ video và nhấn 'q' để thoát demo.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở camera index {CAMERA_INDEX}")
    else:
        # THIẾT LẬP ĐỘ PHÂN GIẢI (CỰC KỲ QUAN TRỌNG: Phải khớp với lúc chụp ảnh calib)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Lấy độ phân giải thực tế của camera
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Camera Live Resolution: {actual_w}x{actual_h}")
        if actual_w != img_shape[0] or actual_h != img_shape[1]:
            print(f"[WARNING] Độ phân giải demo ({actual_w}x{actual_h}) KHÔNG KHỚP với ảnh Calib ({img_shape[0]}x{img_shape[1]})!")
            print(" -> Kết quả khử méo sẽ bị sai/méo hơn!")
        
        # Tính toán ma trận camera mới và bản đồ (map) để khử méo nhanh
        # alpha=0: Zoom vào để xóa viền đen
        # alpha=1: Giữ nguyên toàn bộ ảnh (có viền đen)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (actual_w, actual_h), 0, (actual_w, actual_h)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, new_camera_matrix,
            (actual_w, actual_h), cv2.CV_16SC2
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Khử méo bằng remap (nhanh hơn undistort)
            undist_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            
            # Hiển thị
            # Resize để hiển thị nếu quá to so với màn hình
            h_disp, w_disp = frame.shape[:2]
            if w_disp > 800:
                scale = 800 / w_disp
                frame_disp = cv2.resize(frame, (int(w_disp * scale), int(h_disp * scale)))
                undist_disp = cv2.resize(undist_frame, (int(undist_frame.shape[1] * scale), int(undist_frame.shape[0] * scale)))
            else:
                frame_disp = frame
                undist_disp = undist_frame

            cv2.imshow("LIVE: Original (Before)", frame_disp)
            cv2.imshow("LIVE: Undistorted (After)", undist_disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Đã đóng Live Demo.")

    # -------- TÍNH RE-PROJECTION ERROR CHI TIẾT --------
    print("\n" + "=" * 60)
    print("  RE-PROJECTION ERROR CHI TIẾT CHO TỪNG ẢNH")
    print("=" * 60)

    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                               camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        total_error += error
        print(f"  Ảnh {i + 1:3d}: error = {error:.4f} pixel")

    mean_error = total_error / len(objpoints)
    print(f"\n  Mean re-projection error: {mean_error:.4f} pixel")
    print("\n HOÀN TẤT CALIBRATION!")


if __name__ == "__main__":
    main()
