from ultralytics import YOLO
import cv2 as cv
from collections import deque, defaultdict
import numpy as np
import time


def process_and_read_plate(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bilateral_filtered = cv.bilateralFilter(gray, 8, 75, 75)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv.filter2D(bilateral_filtered, -1, kernel)


    preprocessed_img = cv.cvtColor(sharpened, cv.COLOR_GRAY2BGR)
    results = ocr(preprocessed_img, conf=0.3, verbose=False)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                   'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                   'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z']


    characters = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_index = int(box.cls[0])
            confidence = box.conf[0].item()
            cls = class_names[cls_index]

            # Add the characters and their coordinates to the list.
            characters.append((x1, y1, cls, confidence))

            #Bounding Boxes
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.rectangle(img, (x1, y1 - 20), (x2, y1), (255, 0, 0), cv.FILLED)
            cv.putText(img, cls, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Koordinatlara göre sıralama (x1'e göre artan sırada)
    characters_sorted = sorted(characters, key=lambda char: char[0])

    # Sort by coordinates (in ascending order of x1).
    plate_text = ''.join([char[2] for char in characters_sorted])

    # print
    #print(f"Recognized Plate: {plate_text}")

    return img, plate_text

def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        aspect_ratio = height / float(h)
        new_dimensions = (int(w * aspect_ratio), height)
    else:
        aspect_ratio = width / float(w)
        new_dimensions = (int(width), int(h * aspect_ratio))
    resized_image = cv.resize(image, new_dimensions, interpolation=inter)
    return resized_image


# Load models
car_dedector = YOLO("yolov8s.pt")
plate_dedector = YOLO("plate_dedector.pt")
ocr = YOLO("ocr.pt")

# Declare vehicle_indexes
vehicle_indexes = {2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}
cap = cv.VideoCapture("highway1.avi")
fps = cap.get(cv.CAP_PROP_FPS)

# ROI points
area = np.array([[375, 5], [925, 5], [1100, 650], [150, 650]], dtype=np.int32)

# Use deque to keep track of the last 4 unique vehicle IDs
vehicles = deque(maxlen=4)

# Dictionary to keep track of frame counters for each vehicle ID
frame_counters = defaultdict(int)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = resize(frame, 1250)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display real-time elapsed
    frame_real_time = frame_count / fps  #
    elapsed_video_time = time.strftime("%H:%M:%S", time.gmtime(frame_real_time))
    cv.putText(frame, f"Video Time: {elapsed_video_time}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.polylines(frame, [area], isClosed=True, color=(0, 255, 0), thickness=2)
    vehicle_indexes_results = car_dedector.track(frame, persist=True, verbose=False)[0]
    vehicle_indexes_boxes = np.array(vehicle_indexes_results.boxes.data.tolist(), dtype="int")


   # Draw table
    cv.rectangle(frame, (998, 0), (1250, 251), (0, 0, 0), -1)
    cv.rectangle(frame, (1000, 2), (1248, 249), (95, 183, 210), -1)

    cv.line(frame, (1000, 50), (1250, 50), (0, 0, 0), 2)
    cv.line(frame, (1000, 100), (1250, 100), (0, 0, 0), 2)
    cv.line(frame, (1000, 150), (1250, 150), (0, 0, 0), 2)
    cv.line(frame, (1000, 200), (1250, 200), (0, 0, 0), 2)
    cv.line(frame, (1100, 0), (1100, 250), (0, 0, 0), 2)

    cv.putText(frame, "#ID", (1010, 40), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 100, 0), 2)
    cv.putText(frame, "Plate", (1110, 40), cv.FONT_HERSHEY_DUPLEX, 1.2, (0, 100, 0), 2)


    # Create a set to keep track of currently detected IDs
    current_ids = set()

    for vehicle_box in vehicle_indexes_boxes:
        if len(vehicle_box) == 7:
            x1, y1, x2, y2, track_id, score, class_id = vehicle_box
        else:
            x1, y1, x2, y2, score, class_id = vehicle_box
            track_id = None

        center_x = int((x1 + x2) / 2)
        center_y = int(y2)
        is_inside = cv.pointPolygonTest(area, (center_x, center_y), measureDist=False)
        if class_id in vehicle_indexes and is_inside == 1:
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.rectangle(frame, (center_x - 50, center_y), (center_x + 50, y2 + 35), (255, 0, 0), -1)
            cv.putText(frame, f"#Id:{track_id}", (center_x - 50, center_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 0), 2)

            if track_id is not None:
                current_ids.add(track_id)

            # Extract the region of interest (ROI) for license plate detection
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue  # Skip if ROI is empty

            plate_results = plate_dedector.predict(source=roi, verbose=False)[0]
            plate_boxes = np.array(plate_results.boxes.data.tolist(), dtype="int")

            for plate_box in plate_boxes:
                if len(plate_box) == 6:
                    x1p, y1p, x2p, y2p, score, class_id = plate_box
                    cv.rectangle(frame, (x1 + x1p, y1 + y1p), (x1 + x2p, y1 + y2p), (0, 255, 0), 2)

                    # Associate a track_id with each detected license plate
                    plate_id = track_id if track_id is not None else "Unknown"
                    cv.putText(frame, f"Plate ID: {plate_id}", (x1 + x1p, y1 + y2p + 20), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 0), 2)


                    plate_roi = roi[y1p:y2p, x1p:x2p]

                    # Only process the plate after 10 frames
                    if frame_counters[track_id] >= 10:
                        # Process and read the plate before resizing
                        processed_plate, plate_text = process_and_read_plate(plate_roi)

                        # Scale plate
                        plate_roi_resized = resize(processed_plate, width=int(processed_plate.shape[1] * 3))
                        readed, plate_text = process_and_read_plate(plate_roi_resized)


                        plate_height, plate_width = readed.shape[:2]

                        # Locate scaled plate
                        top_left_corner_x = x1 + x1p
                        top_left_corner_y = y1 + y1p - plate_height

                        # Control plate location
                        if top_left_corner_y < 0:
                            top_left_corner_y = 0

                        if (top_left_corner_y + plate_height <= frame.shape[0] and
                                top_left_corner_x + plate_width <= frame.shape[1]):
                            frame[top_left_corner_y:top_left_corner_y + plate_height,
                            top_left_corner_x:top_left_corner_x + plate_width] = plate_roi_resized
                    else:
                        frame_counters[track_id] += 1

    # Update vehicles list with new IDs, maintaining only the last 4 unique IDs
    vehicles.extend([vid for vid in current_ids if vid not in vehicles])
    if len(vehicles) > 4:
        vehicles = list(vehicles)[-4:]

    # Draw vehicle IDs in the table
    for i, vehicle_id in enumerate(reversed(vehicles)):
        cv.putText(frame, f"{vehicle_id}", (1010, 90 + i * 50), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 2)

    cv.imshow("2", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
