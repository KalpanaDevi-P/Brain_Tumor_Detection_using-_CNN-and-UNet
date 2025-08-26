import os
import cv2
import numpy as np
from tqdm import tqdm
import imutils
import pandas as pd
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

def crop_image(image):
    """
    Crop the region of interest (tumor area) from an MRI image 
    by detecting the largest contour.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Simple thresholding to separate tumor region
    _, threshold = cv2.threshold(gray_image, 45, 255, cv2.THRESH_BINARY)
    threshold = cv2.erode(threshold, None, iterations=2)
    threshold = cv2.dilate(threshold, None, iterations=2)

    # Find the largest contour
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return image  # return original if no contour found

    contour = max(contours, key=cv2.contourArea)

    # Extreme points
    left = tuple(contour[contour[:, :, 0].argmin()][0])
    right = tuple(contour[contour[:, :, 0].argmax()][0])
    top = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    cropped = image[top[1]: bottom[1], left[0]: right[0]]

    return cropped


'''def preprocess_images(base_folder="archive", output_folder="finalim", target_size=224):
    """
    Preprocess all MRI images from Training and Testing sets:
    - Crop to region of interest
    - Resize to target size
    - Save to output directory
    """
    for dataset in ["Training", "Testing"]:
        dataset_path = os.path.join(base_folder, dataset)

        if not os.path.exists(dataset_path):
            print(f"Dataset folder not found: {dataset_path}")
            continue

        for category in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category)
            save_path = os.path.join(output_folder, dataset, category)

            os.makedirs(save_path, exist_ok=True)

            for image_file in tqdm(os.listdir(category_path), desc=f"Processing {dataset}/{category}"):
                image_path = os.path.join(category_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Skipped invalid file: {image_path}")
                    continue

                cropped_image = crop_image(image)
                resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

                save_file_path = os.path.join(save_path, image_file)
                cv2.imwrite(save_file_path, resized_image)


if __name__ == "__main__":
    preprocess_images()'''
def preprocess_and_save_csv(base_folder="archive", output_folder="finalim", target_size=224):
    """
    Preprocess MRI images and save two separate CSV files:
    - train_labels.csv
    - test_labels.csv
    """
    for dataset in ["Training", "Testing"]:
        records = []
        dataset_path = os.path.join(base_folder, dataset)

        if not os.path.exists(dataset_path):
            print(f"Dataset folder not found: {dataset_path}")
            continue

        for category in os.listdir(dataset_path):
            if category not in CATEGORIES:
                continue
            label = CATEGORIES.index(category)  # numeric label

            category_path = os.path.join(dataset_path, category)
            save_path = os.path.join(output_folder, dataset, category)
            os.makedirs(save_path, exist_ok=True)

            for image_file in tqdm(os.listdir(category_path), desc=f"Processing {dataset}/{category}"):
                image_path = os.path.join(category_path, image_file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Skipped invalid file: {image_path}")
                    continue

                cropped_image = crop_image(image)
                resized_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

                save_file_path = os.path.join(save_path, image_file)
                cv2.imwrite(save_file_path, resized_image)

                # Append record for CSV
                records.append([save_file_path, label])

        # Save CSV for this dataset
        csv_file = f"{dataset.lower()}_labels.csv"  # train_labels.csv / test_labels.csv
        csv_path = os.path.join(output_folder, csv_file)
        df = pd.DataFrame(records, columns=["image_path", "label"])
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved {len(records)} images and CSV: {csv_path}")

if __name__ == "__main__":
    preprocess_and_save_csv()