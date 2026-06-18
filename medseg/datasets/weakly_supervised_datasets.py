"""Weakly supervised segmentation datasets.

Supports:
- Box-supervised segmentation (bounding box annotations)
- Image-level classification labels only
- CAM-based weak supervision
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, List, Tuple, Dict, Any


class WeaklySupervisedDataset(Dataset):
    """Base weakly supervised dataset.
    
    Supports multiple types of weak supervision:
    - Bounding boxes
    - Image-level labels
    - Points
    - Scribbles
    
    Args:
        image_dir: Directory containing images
        annotation_file: JSON file with weak annotations
        supervision_type: Type of weak supervision ('box', 'image_label', 'point', 'scribble')
        transform: Data transform
        img_size: Target image size
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        supervision_type: str = 'box',
        transform=None,
        img_size: int = 224,
        num_classes: int = 5,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.supervision_type = supervision_type
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = num_classes
        
        # Load annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # Validate supervision type
        assert supervision_type in ['box', 'image_label', 'point', 'scribble'], \
            f"Unknown supervision type: {supervision_type}"
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        image = self._load_image(ann['image'])
        
        # Load weak annotations based on type
        if self.supervision_type == 'box':
            boxes, box_classes = self._load_boxes(ann)
            image_labels = self._boxes_to_image_labels(box_classes)
            return {
                'image': image,
                'boxes': boxes,          # (N, 4) variable length, no padding
                'box_classes': box_classes,  # (N,) per-box class indices
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'image_label':
            image_labels = self._load_image_labels(ann)
            return {
                'image': image,
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'point':
            points, point_classes = self._load_points(ann)
            image_labels = self._points_to_image_labels(point_classes)
            return {
                'image': image,
                'points': points,              # (N, 2) variable length
                'point_classes': point_classes,  # (N,) per-point class indices
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
        elif self.supervision_type == 'scribble':
            scribbles, scribble_classes = self._load_scribbles(ann)
            image_labels = self._scribbles_to_image_labels(scribble_classes)
            return {
                'image': image,
                'scribbles': scribbles,              # (N, 2) flattened points
                'scribble_classes': scribble_classes,  # (M,) per-scribble class indices
                'image_labels': image_labels,
                'case_name': ann.get('case_name', os.path.basename(ann['image']))
            }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load image."""
        full_path = os.path.join(self.image_dir, path)
        image = Image.open(full_path).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        if self.transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.transform({"image": image, "label": dummy_mask})
            image = sample["image"]
            # Transpose HWC -> CHW after transform (matching GenericDataset)
            if isinstance(image, np.ndarray):
                image = np.ascontiguousarray(image.transpose(2, 0, 1))
                image = torch.from_numpy(image).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image
    
    def _load_boxes(self, ann: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load bounding boxes and their class indices.
        
        Supports two JSON formats:
          1. Simple list: {"boxes": [[x1,y1,x2,y2], ...]}  (all class 0)
          2. Per-box class: {"boxes": [{"box": [x1,y1,x2,y2], "class": c}, ...]}
        
        Coordinates are normalized (0~1) and scaled to image size.
        
        Returns:
            boxes: (N, 4) tensor of [x1,y1,x2,y2] in pixel coords
            box_classes: (N,) tensor of per-box class indices
        """
        boxes_raw = ann.get('boxes', [])
        if len(boxes_raw) == 0:
            return torch.empty(0, 4), torch.empty(0, dtype=torch.long)
        
        boxes_list = []
        classes_list = []
        
        for item in boxes_raw:
            if isinstance(item, dict):
                # Per-box class format
                box = item['box']
                cls = item.get('class', 0)
            elif isinstance(item, (list, tuple)):
                # Simple format — assume class 0
                box = item
                cls = 0
            else:
                continue
            
            boxes_list.append(box)
            classes_list.append(cls)
        
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        box_classes = torch.tensor(classes_list, dtype=torch.long)
        
        # Scale normalized coords to image size
        boxes[:, 0] = boxes[:, 0] * self.img_size[0]  # x1
        boxes[:, 1] = boxes[:, 1] * self.img_size[1]  # y1
        boxes[:, 2] = boxes[:, 2] * self.img_size[0]  # x2
        boxes[:, 3] = boxes[:, 3] * self.img_size[1]  # y2
        
        return boxes, box_classes
    
    def _boxes_to_image_labels(self, box_classes: torch.Tensor) -> torch.Tensor:
        """Convert per-box class indices to image-level labels.
        
        This is the instance-to-semantic conversion: each box contributes
        its class to the image-level label set.
        
        Args:
            box_classes: (N,) tensor of per-box class indices.
        
        Returns:
            image_labels: (num_classes,) tensor with 1.0 for present classes.
        """
        image_labels = torch.zeros(self.num_classes)
        if box_classes.numel() > 0:
            for cls in box_classes.tolist():
                if 0 <= cls < self.num_classes:
                    image_labels[cls] = 1.0
        return image_labels
    
    def _load_image_labels(self, ann: dict) -> torch.Tensor:
        """Load image-level labels."""
        labels = ann.get('image_labels', [])
        image_labels = torch.zeros(self.num_classes)
        
        if isinstance(labels, list):
            for label in labels:
                if isinstance(label, int):
                    image_labels[label] = 1.0
                elif isinstance(label, dict):
                    class_id = label.get('class', 0)
                    image_labels[class_id] = 1.0
        elif isinstance(labels, dict):
            for class_id, present in labels.items():
                if present:
                    image_labels[int(class_id)] = 1.0
        
        return image_labels
    
    def _load_points(self, ann: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load point annotations and their class indices.
        
        Supports two JSON formats:
          1. Simple list: {"points": [[x,y,class], ...]}
          2. Per-point class: {"points": [{"point": [x,y], "class": c}, ...]}
        
        Coordinates are normalized (0~1) and scaled to image size.
        
        Returns:
            points: (N, 2) tensor of [x,y] in pixel coords
            point_classes: (N,) tensor of per-point class indices
        """
        points_raw = ann.get('points', [])
        if len(points_raw) == 0:
            return torch.empty(0, 2), torch.empty(0, dtype=torch.long)
        
        points_list = []
        classes_list = []
        
        for item in points_raw:
            if isinstance(item, dict):
                pt = item['point']
                cls = item.get('class', 0)
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                # Simple format: [x, y, class]
                pt = [item[0], item[1]]
                cls = item[2]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # No class info — default class 0
                pt = item
                cls = 0
            else:
                continue
            
            points_list.append(pt)
            classes_list.append(cls)
        
        points = torch.tensor(points_list, dtype=torch.float32)
        point_classes = torch.tensor(classes_list, dtype=torch.long)
        
        # Scale normalized coords to image size
        points[:, 0] = points[:, 0] * self.img_size[0]  # x
        points[:, 1] = points[:, 1] * self.img_size[1]  # y
        
        return points, point_classes
    
    def _points_to_image_labels(self, point_classes: torch.Tensor) -> torch.Tensor:
        """Convert per-point class indices to image-level labels.
        
        Args:
            point_classes: (N,) tensor of per-point class indices.
        
        Returns:
            image_labels: (num_classes,) tensor with 1.0 for present classes.
        """
        image_labels = torch.zeros(self.num_classes)
        if point_classes.numel() > 0:
            for cls in point_classes.tolist():
                if 0 <= cls < self.num_classes:
                    image_labels[cls] = 1.0
        return image_labels
    
    def _load_scribbles(self, ann: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load scribble annotations and their class indices.
        
        Supports two JSON formats:
          1. Simple list: {"scribbles": [[x,y], [x,y], ...]}  (all class 0)
          2. Per-scribble class: {"scribbles": [{"scribble": [[x,y],...], "class": c}, ...]}
        
        Coordinates are normalized (0~1) and scaled to image size.
        
        Returns:
            scribbles: (N, 2) tensor of all [x,y] points (all scribbles flattened)
            scribble_classes: (M,) tensor of per-scribble class indices
                (M = number of scribbles, N = total points across all scribbles)
        """
        scribbles_raw = ann.get('scribbles', [])
        if len(scribbles_raw) == 0:
            return torch.empty(0, 2), torch.empty(0, dtype=torch.long)
        
        points_list = []
        classes_list = []
        
        for item in scribbles_raw:
            if isinstance(item, dict):
                pts = item['scribble']
                cls = item.get('class', 0)
            elif isinstance(item, (list, tuple)):
                # Simple format: [[x,y], [x,y], ...]
                pts = item
                cls = 0
            else:
                continue
            
            for pt in pts:
                points_list.append(pt)
            classes_list.append(cls)
        
        scribbles = torch.tensor(points_list, dtype=torch.float32)
        scribble_classes = torch.tensor(classes_list, dtype=torch.long)
        
        if scribbles.numel() > 0:
            # Scale normalized coords to image size
            scribbles[:, 0] = scribbles[:, 0] * self.img_size[0]
            scribbles[:, 1] = scribbles[:, 1] * self.img_size[1]
        
        return scribbles, scribble_classes
    
    def _scribbles_to_image_labels(self, scribble_classes: torch.Tensor) -> torch.Tensor:
        """Convert per-scribble class indices to image-level labels.
        
        Args:
            scribble_classes: (M,) tensor of per-scribble class indices.
        
        Returns:
            image_labels: (num_classes,) tensor with 1.0 for present classes.
        """
        image_labels = torch.zeros(self.num_classes)
        if scribble_classes.numel() > 0:
            for cls in scribble_classes.tolist():
                if 0 <= cls < self.num_classes:
                    image_labels[cls] = 1.0
        return image_labels


class BoxSupervisedDataset(WeaklySupervisedDataset):
    """Dataset for box-supervised segmentation.
    
    Only bounding box annotations required.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['supervision_type'] = 'box'
        super().__init__(*args, **kwargs)


class ImageLabelDataset(WeaklySupervisedDataset):
    """Dataset for image-level label supervision.
    
    Only image-level classification labels required.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs['supervision_type'] = 'image_label'
        super().__init__(*args, **kwargs)


class CAMDataset(Dataset):
    """Dataset for CAM-based weak supervision.
    
    Uses pre-computed Class Activation Maps.
    
    Args:
        image_dir: Directory with images
        cam_dir: Directory with CAM files
        label_file: File with image-level labels
        transform: Data transform
        img_size: Target image size
        num_classes: Number of classes
    """
    
    def __init__(
        self,
        image_dir: str,
        cam_dir: str,
        label_file: str,
        transform=None,
        img_size: int = 224,
        num_classes: int = 5,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.cam_dir = cam_dir
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = num_classes
        
        # Load labels
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # Get image list
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load image
        image = self._load_image(img_file)
        
        # Load CAM
        cam_file = os.path.splitext(img_file)[0] + '.npy'
        cam_path = os.path.join(self.cam_dir, cam_file)
        cam = self._load_cam(cam_path)
        
        # Load image-level labels
        image_labels = self._get_image_labels(img_file)
        
        return {
            'image': image,
            'cam': cam,
            'image_labels': image_labels,
            'case_name': img_file
        }
    
    def _load_image(self, filename: str) -> torch.Tensor:
        """Load image."""
        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        
        if self.transform is not None:
            dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
            sample = self.transform({"image": image, "label": dummy_mask})
            image = sample["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image
    
    def _load_cam(self, path: str) -> torch.Tensor:
        """Load CAM."""
        if os.path.exists(path):
            cam = np.load(path)
        else:
            # Generate dummy CAM
            cam = np.random.rand(self.num_classes, self.img_size[0], self.img_size[1]).astype(np.float32)
        
        if isinstance(cam, np.ndarray):
            cam = torch.from_numpy(cam).float()
        
        return cam
    
    def _get_image_labels(self, filename: str) -> torch.Tensor:
        """Get image-level labels."""
        image_labels = torch.zeros(self.num_classes)
        
        if filename in self.labels:
            labels = self.labels[filename]
            if isinstance(labels, list):
                for label in labels:
                    image_labels[label] = 1.0
            elif isinstance(labels, dict):
                for class_id, present in labels.items():
                    if present:
                        image_labels[int(class_id)] = 1.0
        
        return image_labels
