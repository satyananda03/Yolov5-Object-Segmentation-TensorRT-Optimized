import cv2 
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5n_4Class.engine", conf=0.1, yolo_ver="v5")

# Set pipeline CSI Camera
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30
    #flip_method=0,
):
     return (
         f"nvarguscamerasrc sensor-id={sensor_id} ! "
         f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
         f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
         f"nvvidconv ! "
         f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
         f"videoconvert ! video/x-raw, format=(string)BGR ! "
         f"appsink max-buffers=1 drop=True sync=false"
     )

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    model.Inference(frame)
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
