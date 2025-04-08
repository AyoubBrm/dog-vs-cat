from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-cls.pt")
    results = model.train(data="C:\\Users\\ayoub\\Desktop\\dog&cat", epochs=100, imgsz=64, device=0)