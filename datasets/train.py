from ultralytics import YOLO

if __name__ == '__main__':

    # load YOLO V8 model
    model = YOLO('yolov8n.pt')

    # train on basketball dataset
    results = model.train(data='basketball.yaml', epochs=100, imgsz=640, device='mps')