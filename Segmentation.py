import serial
import time
import cv2
from yoloSeg import YoloTRTSeg

# Load model YOLO TensorRT
model = YoloTRTSeg(library="yolov5/build/libmyplugins.so",
                engine_path="yolov5/build/yolov5n-seg_640.engine",
                conf=0.1, yolo_ver="v5")

# Fungsi pipeline CSI Camera (sama seperti sebelumnya)
def gstreamer_pipeline(sensor_id=0, capture_width=640, capture_height=480,
                       display_width=640, display_height=480, framerate=30):
    return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1, format=NV12 ! "
            "nvvidconv flip-method=0 ! queue max-size-buffers=1 leaky=downstream ! "
            "video/x-raw, width=%d, height=%d, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true max-lateness=0"
            % (sensor_id, capture_width, capture_height, framerate, display_width, display_height)
    )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferensi
    result, time = model.Inference(frame)
    vis = result["vis_image"]

    # Tampilkan
    cv2.putText(vis, f"FPS: {time:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("YoloV5-Seg TensorRT Optimized", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
