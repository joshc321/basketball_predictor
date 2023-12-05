from ultralytics import YOLO

if __name__ == '__main__':

    # load YOLO V8 model
    # model = YOLO('yolov8n.pt')
    model = YOLO('/Users/joshcordero/Code/School/cs117/final_project/datasets/runs/detect/train3/weights/last.pt')
    
    # train on basketball dataset
    # results = model.train(data='basketball.yaml', epochs=100, imgsz=640, device='mps')
    results = model.train(resume=True, device='mps')