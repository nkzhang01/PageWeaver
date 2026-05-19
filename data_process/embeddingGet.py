from pathlib import Path
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoProcessor
import torch.nn.functional as F

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def clear_dir_contents(dir_path: Path):
    if not dir_path.exists():
        return
    for p in dir_path.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


class UIEmbeddingExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        sig_id = "google/siglip2-so400m-patch16-naflex"
        self.sig_model = AutoModel.from_pretrained(sig_id).to(self.device).eval()
        self.sig_proc = AutoProcessor.from_pretrained(sig_id)

        dino_id = "facebook/dinov2-base"
        self.dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained(dino_id)

    def _extract_edge_img(self, img_path: Path) -> Image.Image:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edge_rgb = cv2.merge([edges, edges, edges])
        return Image.fromarray(edge_rgb)

    @torch.no_grad()
    def get_features(self, img_path: Path):
        raw_img = Image.open(img_path).convert("RGB")
        sig_inputs = self.sig_proc(images=raw_img, return_tensors="pt").to(self.device)
        sig_feat = self.sig_model.get_image_features(**sig_inputs)
        sig_feat = F.normalize(sig_feat, p=2, dim=-1)

        edge_img = self._extract_edge_img(img_path)
        dino_inputs = self.dino_proc(images=edge_img, return_tensors="pt").to(self.device)
        dino_out = self.dino_model(**dino_inputs)
        dino_feat = dino_out.last_hidden_state[:, 0, :]
        dino_feat = F.normalize(dino_feat, p=2, dim=-1)

        combined_feat = torch.cat([sig_feat, dino_feat], dim=-1)

        semantic = sig_feat.cpu().numpy().astype(np.float32).flatten()
        layout = dino_feat.cpu().numpy().astype(np.float32).flatten()
        combined = combined_feat.cpu().numpy().astype(np.float32).flatten()
        return semantic, layout, combined


def save_embeddings_for_dir(app_dir: Path, extractor: UIEmbeddingExtractor):
    img_paths = sorted([p for p in app_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])

    if not img_paths:
        return 0

    out_dir = app_dir / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_dir_contents(out_dir)

    print(f"Extracting embeddings for {len(img_paths)} images in: {app_dir}")

    index_rows = []
    for img_path in tqdm(img_paths):
        semantic, layout, combined = extractor.get_features(img_path)

        out_path = out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(
            out_path,
            image_name=img_path.name,
            image_path=str(img_path),
            semantic=semantic,
            layout=layout,
            combined=combined,
        )

        index_rows.append(
            {
                "image_name": img_path.name,
                "npz_path": str(out_path),
                "semantic_dim": int(semantic.shape[0]),
                "layout_dim": int(layout.shape[0]),
                "combined_dim": int(combined.shape[0]),
            }
        )

    # Save a global index for easy loading
    index_path = out_dir / "index.npz"
    image_names = np.array([r["image_name"] for r in index_rows], dtype=object)
    npz_paths = np.array([r["npz_path"] for r in index_rows], dtype=object)
    semantic_dims = np.array([r["semantic_dim"] for r in index_rows], dtype=np.int32)
    layout_dims = np.array([r["layout_dim"] for r in index_rows], dtype=np.int32)
    combined_dims = np.array([r["combined_dim"] for r in index_rows], dtype=np.int32)
    np.savez_compressed(
        index_path,
        image_names=image_names,
        npz_paths=npz_paths,
        semantic_dims=semantic_dims,
        layout_dims=layout_dims,
        combined_dims=combined_dims,
    )

    print(f"Done. Per-image embeddings saved to: {out_dir}")
    print(f"Global index saved to: {index_path}")
    return len(img_paths)


def find_app_dirs(data_root: Path):
    app_dirs = set()
    for img_path in data_root.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        if "embeddings" in img_path.parts:
            continue
        app_dirs.add(img_path.parent)
    return sorted(app_dirs)


def save_embeddings_for_all_apps(data_root: Path):
    app_dirs = find_app_dirs(data_root)
    if not app_dirs:
        print(f"No app image directories found under: {data_root}")
        return

    print(f"Found {len(app_dirs)} app directories under: {data_root}")
    extractor = UIEmbeddingExtractor()

    total_images = 0
    processed_apps = 0
    for app_dir in app_dirs:
        count = save_embeddings_for_dir(app_dir, extractor)
        if count > 0:
            total_images += count
            processed_apps += 1

    print(f"Finished. Processed app directories: {processed_apps}, total images: {total_images}")


if __name__ == "__main__":
    data_root = Path(r"D:\ui2code\InteractionSequence\data")
    save_embeddings_for_all_apps(data_root)
