from ultralytics import YOLO

model = YOLO(r"prediction_pipeline\best_model_8.pt")
result = model.predict(r"dataset_resized\20250516_162705.jpg")
