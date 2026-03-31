import os, csv
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

DEFAULT_CLASS_MAP = {
    "akiec": 1,  # Actinic keratoses / intraepithelial carcinoma
    "bcc":   2,  # Basal cell carcinoma
    "bkl":   3,  # Benign keratosis-like lesions
    "df":    4,  # Dermatofibroma
    "nv":    5,  # Melanocytic nevi
    "mel":   6,  # Melanoma
    "vasc":  7,  # Vascular lesions
}

# Clinical severity scores per class name (higher = more dangerous / urgent).
DEFAULT_SEVERITY_MAP = {
    "akiec": 3,
    "bcc":   3,
    "bkl":   1,
    "df":    1,
    "nv":    1,
    "mel":   4,   # melanoma: highest priority
    "vasc":  2,
}

def _norm_name(s: str) -> str:
    return s.strip().lower()

def det_collate(batch):
    imgs, tgts = list(zip(*batch))
    return list(imgs), list(tgts)

class HAM10000Detection(Dataset):
    """
    HAM10000 object-detection dataset (bounding boxes).
    - Returns (image_tensor, target_dict) per torchvision detection API.
    - target['labels'] are in 1..K (background is implicit).
    """
    def __init__(
        self,
        images_dir: str,
        annotations_csv: str,
        class_map: Optional[Dict[str, int]] = None,
        image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        transforms=None,
    ):
        self.images_dir = images_dir
        self.annotations_csv = annotations_csv
        self.class_map = class_map or DEFAULT_CLASS_MAP
        self.transforms = transforms
        self.image_exts = image_exts

        self._idx_to_image: List[str] = []
        self._image_to_ann: Dict[str, List[Tuple[List[float], int]]] = {}

        self._read_csv_build_index()

        # num classes (K)
        self.num_classes = max(self.class_map.values())

    def _read_csv_build_index(self):
        with open(self.annotations_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # columns (case-insensitive)
                cols = {k.lower(): v for k, v in row.items()}

                img_id = cols.get("image_id") or cols.get("image") or cols.get("file") or cols.get("filename")
                assert img_id is not None, "CSV must contain image_id/image/file/filename"

                # strip extension for consistent lookup
                root, ext = os.path.splitext(img_id)
                img_key = root  # key without extension

                # boxes can be (x_min,y_min,x_max,y_max) or (x_min,y_min,width,height)
                def _to_float(name, default=None):
                    v = cols.get(name)
                    return float(v) if v not in (None, "",) else default

                xmin = _to_float("x_min", _to_float("xmin"))
                ymin = _to_float("y_min", _to_float("ymin"))
                xmax = _to_float("x_max", _to_float("xmax"))
                ymax = _to_float("y_max", _to_float("ymax"))
                width = _to_float("width", _to_float("w"))
                height = _to_float("height", _to_float("h"))

                assert xmin is not None and ymin is not None, "Need x_min and y_min in CSV"
                if xmax is None or ymax is None:
                    assert width is not None and height is not None, "Provide x_max/y_max or width/height"
                    xmax = xmin + width
                    ymax = ymin + height

                # label can be string (dx) or numeric id
                label_id = None
                if cols.get("label_id") is not None:
                    label_id = int(cols["label_id"])
                else:
                    label_str = cols.get("label") or cols.get("dx") or cols.get("class")
                    assert label_str is not None, "Need label / dx / class in CSV"
                    label_id = self.class_map[_norm_name(label_str)]

                box = [xmin, ymin, xmax, ymax]
                self._image_to_ann.setdefault(img_key, []).append((box, label_id))

        # Stable index of images (keys sorted)
        self._idx_to_image = sorted(self._image_to_ann.keys())

    def __len__(self) -> int:
        return len(self._idx_to_image)

    def _find_image_path(self, key_no_ext: str) -> str:
        # Try common extensions
        for ext in self.image_exts:
            p = os.path.join(self.images_dir, key_no_ext + ext)
            if os.path.exists(p):
                return p
        # If not found, scan dir (fallback)
        for name in os.listdir(self.images_dir):
            if name.startswith(key_no_ext):
                return os.path.join(self.images_dir, name)
        raise FileNotFoundError(f"Image for key '{key_no_ext}' not found with extensions {self.image_exts}")

    def __getitem__(self, idx: int):
        key = self._idx_to_image[idx]
        img_path = self._find_image_path(key)
        img = Image.open(img_path).convert("RGB")

        ann = self._image_to_ann[key]
        boxes = torch.tensor([a[0] for a in ann], dtype=torch.float32)   # (N,4)
        labels = torch.tensor([a[1] for a in ann], dtype=torch.int64)    # (N,) in 1..K

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}

        # Optional transforms: if you plug in box-aware transforms, update both img & boxes.
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        img = pil_to_tensor(img).float() / 255.0

        return img, target

    # ----- helpers for counts/metadata (no image loading) ---------------------
    def class_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for ann_list in self._image_to_ann.values():
            for _, lab in ann_list:
                counts[lab] = counts.get(lab, 0) + 1
        return counts

    def class_counts_tensor(self) -> torch.Tensor:
        counts = torch.zeros(self.num_classes, dtype=torch.long)
        cc = self.class_counts()
        for k, v in cc.items():
            counts[k - 1] = v
        return counts
