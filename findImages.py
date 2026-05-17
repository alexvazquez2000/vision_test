import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

#based on https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/index
#https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python

# Define common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

def find_images(root_dir):
    image_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(dirpath, filename)
                image_files.append(full_path)

    return image_files

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("", img)
  #cv2.waitKey(0)


    
if __name__ == "__main__":
    
    model_path = "efficientnet_lite0.tflite"
    
    #directory = input("Enter directory to search: ").strip()
    directory = "C:\\Users\\alexv\\workspace"
    if not os.path.isdir(directory):
        print("Invalid directory.")
    else:
        images = find_images(directory)
        print(f"\nFound {len(images)} image(s):\n")
        for img_name in images:
            print(img_name)
            image = cv2.imread(img_name)
            resize_and_show(image)

    cv2.destroyAllWindows()

            