from ultralytics import YOLO
import cv2
import numpy as np
weburl = 0
cap = cv2.VideoCapture(weburl)
count = 0
folder_path = "test_dataset/4"
model = YOLO("yolo11n.pt")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Inference

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if model.names[cls] == 'person':
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Ensure coordinates are within frame dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Crop the person
                re_frame = frame[y1:y2, x1:x2]

                if re_frame.size == 0:
                    continue  # Skip if crop is empty

                frame = re_frame

    # Show the original detection frame
    frame = cv2.resize(frame,(64,64))
    cv2.imshow("Detection", frame)
    
	# Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("s"):
        count += 1
        cv2.imwrite(f"{folder_path}/test_image_{count}.jpeg",frame)
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()