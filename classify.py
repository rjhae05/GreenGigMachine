import cv2
import numpy as np
import tensorflow as tf
import argparse

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(image):
    img = cv2.resize(image, (224, 224))  # adjust size if model requires
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    return labels[class_id], confidence

def run_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, confidence = classify_image(frame)
        text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Plastic Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def run_image(image_path):
    image = cv2.imread(image_path)
    label, confidence = classify_image(image)
    print(f"Prediction: {label} ({confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for classification")
    parser.add_argument("--camera", action="store_true", help="Run real-time detection using camera")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    elif args.camera:
        run_camera()
    else:
        print("Please provide --image <path> or --camera")
