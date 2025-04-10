from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n-cls.pt")
    results = model.train(data="./", epochs=100, imgsz=64, device=0)