
# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Name: C J Rohit

## Reg no: 212224243005

## Aim

To write a Python program using OpenCV to perform the following image manipulations:

i) Extract ROI from an image.

ii) Perform face detection using Haar Cascades in static images.

iii) Perform eye detection in images.

iv) Perform face detection with label in real-time video from webcam.

## Software Required

* Anaconda - Python 3.7 or above
* OpenCV library (`opencv-python`)
* Matplotlib library (`matplotlib`)
* Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# PART 1: ROI SEGMENTATION
# =========================

image = cv2.imread('CJPHO.png')

if image is None:
    print("Error: CJPHO.png not found")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# ROI
roi = image[100:420, 200:550]

mask = np.zeros_like(image)
mask[100:420, 200:550] = roi

segmented = cv2.bitwise_and(image, mask)

plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
plt.title("Segmented ROI")
plt.axis('off')
plt.show()


# =========================
# PART 2: EDGE DETECTION
# =========================

image = cv2.imread('CJPHO.png')

if image is None:
    print("Error: CJPHO.png not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

for c in contours:
    if cv2.contourArea(c) > 50:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Contour Detection")
plt.axis('off')
plt.show()


# =========================
# PART 3: OBJECT DETECTION
# =========================

config_file = 'deploy.prototxt'
weights_file = 'mobilenet_iter_73000.caffemodel'

# If model files NOT found, skip safely
if not os.path.exists(config_file) or not os.path.exists(weights_file):
    print("Model files not found - Skipping Object Detection part")

else:
    net = cv2.dnn.readNetFromCaffe(
        config_file,
        weights_file
    )

    class_labels = {
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    image = cv2.imread('CJPHO.png')

    if image is None:
        print("Error: CJPHO.png not found")
        exit()

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = class_labels.get(idx, "Unknown")

            box = detections[0, 0, i, 3:7] * np.array(
                [w, h, w, h]
            )

            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(
                image,
                (startX, startY),
                (endX, endY),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Object Detection (MobileNet-SSD)")
    plt.axis('off')
    plt.show()
```

## Output
<img width="389" height="409" alt="download" src="https://github.com/user-attachments/assets/d6ac9534-bf5d-450a-aab1-ef25e12b4d69" />
<img width="389" height="409" alt="download" src="https://github.com/user-attachments/assets/e77b60af-3d1f-456b-a459-ea8866bfc1ae" />
<img width="389" height="409" alt="download" src="https://github.com/user-attachments/assets/1aa9411a-d87f-44e3-921c-710366eb2008" />
<img width="389" height="409" alt="download" src="https://github.com/user-attachments/assets/b640b07f-1905-467c-b8b5-3e4873f59b81" />
<img width="482" height="502" alt="download" src="https://github.com/user-attachments/assets/535c5888-b6eb-41d2-9bc0-a21fac920535" />

Add the output screenshots obtained after executing the program.

## Result

Thus, the Python program using OpenCV to perform image manipulations for the given objectives was executed successfully.

## Author

**Name:** C J Rohit

**Reg No:** 212224243005
