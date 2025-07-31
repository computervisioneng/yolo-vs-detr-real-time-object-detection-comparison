from ultralytics import YOLO, RTDETR

model_yolo = YOLO("yolo11l.pt")
model_yolo.train(data="config.yaml", epochs=20, imgsz=640, batch=8)

model_rtdetr = RTDETR("rtdetr-l.pt")
model_rtdetr.train(data="config.yaml", epochs=20, imgsz=640, batch=8)
