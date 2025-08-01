import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# --------------------------
# Load ONNX Generator Model
# --------------------------
onnx_model_path = "/content/drive/MyDrive/Colab Notebooks/depthgan_generator.onnx"  # Path to your ONNX model
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

# --------------------------
# Preprocessing Function
# --------------------------
def preprocess_grayscale_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))  # Resize to match ONNX model input
    img = img.astype(np.float32) / 255.0
    img_tensor = transforms.ToTensor()(img).numpy()  # Shape: (1, H, W)
    return img_tensor

# --------------------------
# Inference and Visualization
# --------------------------
def predict_and_visualize(left_path, right_path):
    # Preprocess
    left_tensor = preprocess_grayscale_image(left_path)
    right_tensor = preprocess_grayscale_image(right_path)

    # Add batch dimension -> shape: (1, 1, 256, 256)
    left_tensor = np.expand_dims(left_tensor, axis=0)
    right_tensor = np.expand_dims(right_tensor, axis=0)

    # Run ONNX inference
    inputs = {
        session.get_inputs()[0].name: left_tensor,
        session.get_inputs()[1].name: right_tensor
    }

    output = session.run(None, inputs)
    disparity = output[0][0, 0]  # shape: (256, 256)

    # Compute depth map (depth = (baseline × focal) / disparity)
    baseline_distance = 0.54  # meters (example for KITTI stereo)
    focal_length = 721.0      # pixels (example for KITTI stereo)

    disparity[disparity == 0] = 1e-6  # Avoid divide-by-zero
    depth = (baseline_distance * focal_length) / disparity

    # --------------------------
    # Visualization
    # --------------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    original_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(cv2.resize(original_left, (256, 256)), cmap='gray')
    plt.title("Left Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(disparity, cmap='magma')
    plt.title("Predicted Disparity")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(depth, cmap='inferno')
    plt.title("Estimated Depth")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# --------------------------
# Example Usage
# --------------------------
left_img_path = "/content/drive/MyDrive/Colab Notebooks/image_0/000000.png"
right_img_path = "/content/drive/MyDrive/Colab Notebooks/image_1/000000.png"

predict_and_visualize(left_img_path, right_img_path)
