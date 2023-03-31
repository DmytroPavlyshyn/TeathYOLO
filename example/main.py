from pathlib import Path

from ultralytics import YOLO

root = "/Users/dmytropavlyshyn/PycharmProjects/TeathYOLO"
path = Path(f"{root}/runs/detect/train2/weights/best.pt")
# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(path)  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="/Users/dmytropavlyshyn/Desktop/JSONS_2/YOLODataset/dataset.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
results = model("/Users/dmytropavlyshyn/StudyHNEU/Tufts Dental Database/Radiographs/303.JPG")  # predict on an image
success = model.export()  # export the model to ONNX format

