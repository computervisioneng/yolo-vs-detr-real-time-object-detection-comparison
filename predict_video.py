import time
import os
import shutil

from ultralytics import YOLO, RTDETR
import cv2


video_path = os.path.join('.', 'ducks.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

video_path_out = 'out.mp4'
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = 'model.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

time_ = 0

while ret:

    tic = time.time()
    results = model(frame, imgsz=640, rect=False)[0]
    time_ += (time.time() - tic)
    # print(time.time() - tic)

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

shutil.move(video_path_out, video_path_out.replace('out_', 'out_{}'.format(str(int(time_)))))

out.release()
cap.release()

cv2.destroyAllWindows()
