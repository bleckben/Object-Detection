#!/usr/bin/env python3
"""
YOLO Annotation Tool - Local, Private, No Cloud Required
Annotate images for YOLOv11/YOLOv12 training without using Roboflow or any cloud service.
"""

import cv2
import os
import yaml
import json
from pathlib import Path
import argparse


class YOLOAnnotator:
    def __init__(self, images_dir, labels_dir, classes_file=None, classes_list=None):

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Load classes
        if classes_file and os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
        elif classes_list:
            self.classes = classes_list
        else:
            self.classes = ['object']  # Default class

        # Get all images
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in self.image_extensions
        ])

        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

        self.current_idx = 0
        self.current_image = None
        self.display_image = None
        self.scale = 1.0

        # Annotation state
        self.boxes = []  # List of boxes for current image: [(class_id, x1, y1, x2, y2), ...]
        self.current_box = None
        self.drawing = False
        self.current_class = 0

        # Colors for each class
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        ]

        self.window_name = 'YOLO Annotator - Private & Local'

    def load_image(self, idx):
        """Load image and existing annotations."""
        self.current_idx = idx
        img_path = self.image_files[idx]
        self.current_image = cv2.imread(str(img_path))

        if self.current_image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # Load existing annotations if they exist
        self.boxes = []
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            h, w = self.current_image.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])

                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)

                        self.boxes.append((class_id, x1, y1, x2, y2))

        self.update_display()

    def update_display(self):
        """Update the display with current image and boxes."""
        self.display_image = self.current_image.copy()
        h, w = self.display_image.shape[:2]

        # Draw existing boxes
        for class_id, x1, y1, x2, y2 in self.boxes:
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = self.classes[class_id] if class_id < len(self.classes) else f"Class {class_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(self.display_image, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(self.display_image, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            color = self.colors[self.current_class % len(self.colors)]
            cv2.rectangle(self.display_image, (x1, y1), (x2, y2), color, 2)

        # Add info text
        info = f"Image {self.current_idx + 1}/{len(self.image_files)} | "
        info += f"Class: {self.classes[self.current_class]} ({self.current_class}) | "
        info += f"Boxes: {len(self.boxes)}"
        cv2.putText(self.display_image, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add help text
        help_text = "SPACE:Next | BACKSPACE:Prev | S:Save | D:Delete Last | C:Change Class | Q:Quit"
        cv2.putText(self.display_image, help_text, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.display_image)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_box[2] = x
            self.current_box[3] = y
            self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.current_box:
                x1, y1, x2, y2 = self.current_box

                # Ensure x1 < x2 and y1 < y2
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Only add if box has some area
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.boxes.append((self.current_class, x1, y1, x2, y2))

                self.current_box = None
                self.update_display()

    def save_annotations(self):
        """Save annotations in YOLO format."""
        img_path = self.image_files[self.current_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        h, w = self.current_image.shape[:2]

        with open(label_path, 'w') as f:
            for class_id, x1, y1, x2, y2 in self.boxes:
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Saved: {label_path} ({len(self.boxes)} boxes)")

    def delete_last_box(self):
        """Delete the most recently added box."""
        if self.boxes:
            self.boxes.pop()
            self.update_display()

    def next_image(self):
        """Go to next image."""
        if self.current_idx < len(self.image_files) - 1:
            self.save_annotations()
            self.load_image(self.current_idx + 1)

    def prev_image(self):
        """Go to previous image."""
        if self.current_idx > 0:
            self.save_annotations()
            self.load_image(self.current_idx - 1)

    def change_class(self):
        """Cycle to next class."""
        self.current_class = (self.current_class + 1) % len(self.classes)
        self.update_display()

    def run(self):
        """Start the annotation tool."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.load_image(0)

        print("\n" + "="*70)
        print("YOLO Annotation Tool - Private & Local (No Roboflow Required!)")
        print("="*70)
        print("\nControls:")
        print("  - Draw box: Click and drag with mouse")
        print("  - Next image: SPACE or N")
        print("  - Previous image: BACKSPACE or P")
        print("  - Save: S")
        print("  - Delete last box: D")
        print("  - Change class: C")
        print("  - Quit: Q or ESC")
        print(f"\nClasses: {', '.join(f'{i}:{c}' for i, c in enumerate(self.classes))}")
        print("="*70 + "\n")

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                self.save_annotations()
                break
            elif key == ord(' ') or key == ord('n'):  # SPACE or N
                self.next_image()
            elif key == 8 or key == ord('p'):  # BACKSPACE or P
                self.prev_image()
            elif key == ord('s'):  # S
                self.save_annotations()
            elif key == ord('d'):  # D
                self.delete_last_box()
            elif key == ord('c'):  # C
                self.change_class()

        cv2.destroyAllWindows()
        print("\nAnnotation session completed!")


def setup_dataset_from_images(source_dir, dataset_dir, train_split=0.8):
    import shutil
    from random import shuffle

    source_dir = Path(source_dir)
    dataset_dir = Path(dataset_dir)

    # Create directory structure
    for split in ['train', 'val']:
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    images = [f for f in source_dir.iterdir() if f.suffix.lower() in image_extensions]

    # Shuffle and split
    shuffle(images)
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy images
    for img in train_images:
        shutil.copy(img, dataset_dir / 'images' / 'train' / img.name)

    for img in val_images:
        shutil.copy(img, dataset_dir / 'images' / 'val' / img.name)

    print(f"\nDataset organized:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Total: {len(images)}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Annotation Tool - Private & Local')
    parser.add_argument('--images', type=str, required=True, help='Directory containing images')
    parser.add_argument('--labels', type=str, required=True, help='Directory to save labels')
    parser.add_argument('--classes', type=str, help='Path to classes.txt file (one class per line)')
    parser.add_argument('--class-list', type=str, nargs='+', help='List of classes (e.g., --class-list person car dog)')

    args = parser.parse_args()

    # Determine classes
    classes = None
    if args.class_list:
        classes = args.class_list

    # Create annotator
    annotator = YOLOAnnotator(
        images_dir=args.images,
        labels_dir=args.labels,
        classes_file=args.classes,
        classes_list=classes
    )

    # Run annotation tool
    annotator.run()


if __name__ == '__main__':
    main()
