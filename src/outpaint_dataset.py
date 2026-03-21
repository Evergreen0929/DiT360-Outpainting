import argparse
import io
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_camera_matrices(yaw: float, pitch: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    r_yaw = torch.tensor(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        device=device,
        dtype=torch.float32,
    )

    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    r_pitch = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cos_p, -sin_p], [0.0, sin_p, cos_p]],
        device=device,
        dtype=torch.float32,
    )

    cam_to_world_rot = r_yaw @ r_pitch
    world_to_cam_rot = cam_to_world_rot.T
    return cam_to_world_rot, world_to_cam_rot


def perspective_to_pano_mask_only(
    persp_rgb: torch.Tensor,
    h_fov: float,
    yaw: float,
    pitch: float,
    pano_h: int,
    pano_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = persp_rgb.device
    bsz, _, persp_h, persp_w = persp_rgb.shape

    _, world_to_cam_rot = get_camera_matrices(yaw=yaw, pitch=pitch, device=device)

    v, u = torch.meshgrid(
        torch.linspace(-1, 1, pano_h, device=device),
        torch.linspace(-1, 1, pano_w, device=device),
        indexing="ij",
    )

    theta = u * math.pi
    phi = v * math.pi / 2

    s_x = torch.cos(phi) * torch.sin(theta)
    s_y = torch.sin(phi)
    s_z = -torch.cos(phi) * torch.cos(theta)
    rays_world = torch.stack([s_z, s_y, s_x], dim=-1)

    rays_cam = rays_world.reshape(-1, 3) @ world_to_cam_rot.T
    rays_cam = rays_cam.reshape(pano_h, pano_w, 3)
    cx, cy, cz = rays_cam.unbind(dim=-1)

    mask_cam_front = cz < -1e-8

    h_fov_rad = math.radians(h_fov)
    focal_length = persp_w / (2 * math.tan(h_fov_rad / 2))

    u_proj = focal_length * (cx / -cz)
    v_proj = focal_length * (cy / -cz)

    u_norm = u_proj / (persp_w / 2)
    v_norm = v_proj / (persp_h / 2)
    grid = torch.stack([u_norm, -v_norm], dim=-1)

    valid_sample_mask = mask_cam_front & (u_norm.abs() <= 1) & (v_norm.abs() <= 1)
    grid[~valid_sample_mask] = 2.0

    grid_b = grid.unsqueeze(0).expand(bsz, -1, -1, -1).to(persp_rgb.dtype)
    warped_rgb = F.grid_sample(
        persp_rgb,
        grid_b,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid_mask_b = valid_sample_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).float()
    return warped_rgb, valid_mask_b


def pano_to_perspective(
    pano_rgb: torch.Tensor,
    h_fov: float,
    yaw: float,
    pitch: float,
    persp_h: int,
    persp_w: int,
) -> torch.Tensor:
    device = pano_rgb.device
    bsz, _, pano_h, pano_w = pano_rgb.shape
    if bsz != 1:
        raise ValueError("pano_to_perspective currently expects batch size 1")

    cam_to_world_rot, _ = get_camera_matrices(yaw=yaw, pitch=pitch, device=device)

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, persp_h, device=device),
        torch.linspace(-1, 1, persp_w, device=device),
        indexing="ij",
    )

    h_fov_rad = math.radians(h_fov)
    focal_length = persp_w / (2 * math.tan(h_fov_rad / 2))

    u_proj = xx * (persp_w / 2)
    v_proj = -yy * (persp_h / 2)

    cx = u_proj / focal_length
    cy = v_proj / focal_length
    cz = -torch.ones_like(cx)

    rays_cam = torch.stack([cx, cy, cz], dim=-1)
    rays_cam = rays_cam / rays_cam.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    rays_world = rays_cam.reshape(-1, 3) @ cam_to_world_rot.T
    rays_world = rays_world.reshape(persp_h, persp_w, 3)

    s_z = rays_world[..., 0]
    s_y = rays_world[..., 1]
    s_x = rays_world[..., 2]

    theta = torch.atan2(s_x, -s_z)
    phi = torch.asin(torch.clamp(s_y, -1.0, 1.0))

    u_norm = theta / math.pi
    v_norm = phi / (math.pi / 2)
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).to(pano_rgb.dtype)

    persp_rgb = F.grid_sample(
        pano_rgb,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return persp_rgb


@dataclass
class OutpaintSample:
    target_pixel_values: torch.Tensor
    view_params: torch.Tensor
    num_views: int
    captions: str


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3 uri, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


class RandomPerspOutpaintDataset(Dataset):
    def __init__(
        self,
        id_list_file: str,
        pano_root: str,
        pano_height: int = 512,
        pano_width: int = 1024,
        image_ext: str = "jpg",
        caption_map_file: Optional[str] = None,
        caption_template: str = "This is a panorama image.",
        exclude_prefixes: Optional[List[str]] = None,
        min_views: int = 1,
        max_views: int = 10,
        min_fov: float = 75.0,
        max_fov: float = 105.0,
        min_pitch: float = -30.0,
        max_pitch: float = 30.0,
        perspective_size: int = 512,
    ) -> None:
        self.id_list_file = id_list_file
        self.pano_root = pano_root.rstrip("/")
        self.pano_height = pano_height
        self.pano_width = pano_width
        self.image_ext = image_ext.lstrip(".")
        self.caption_template = caption_template
        self.exclude_prefixes = exclude_prefixes or ["DiT360_"]
        self.min_views = min_views
        self.max_views = max_views
        self.min_fov = min_fov
        self.max_fov = max_fov
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.perspective_size = perspective_size

        self._s3_client = None
        self._caption_map = self._load_caption_map(caption_map_file)
        self.ids = self._load_ids(id_list_file)
        self.subsets = [self._infer_subset_name(x) for x in self.ids]
        self.subset_counts = self._count_subsets(self.subsets)

        self.target_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.pano_height, self.pano_width),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_caption_map(self, caption_map_file: Optional[str]) -> Dict[str, str]:
        if caption_map_file is None:
            return {}
        with open(caption_map_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("caption_map_file must contain a JSON object mapping id -> caption")
        return data

    def _load_ids(self, id_list_file: str) -> List[str]:
        with open(id_list_file, "r", encoding="utf-8") as f:
            ids = json.load(f)
        if not isinstance(ids, list):
            raise ValueError("id list file must be a JSON list")

        valid_ids: List[str] = []
        for item in ids:
            if not isinstance(item, str):
                continue
            if any(item.startswith(prefix) for prefix in self.exclude_prefixes):
                continue
            valid_ids.append(item)

        if not valid_ids:
            raise ValueError("No valid panorama ids left after filtering; check exclude prefixes")
        return valid_ids

    @staticmethod
    def _infer_subset_name(sample_id: str) -> str:
        # Examples: Sun360_xxx, ZInD_xxx, Hunyuan_gen_xxx, Matterport3D_xxx, scene_xxx
        if sample_id.startswith("scene_"):
            return "scene"
        return sample_id.split("_")[0] if "_" in sample_id else sample_id

    @staticmethod
    def _count_subsets(subsets: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in subsets:
            counts[s] = counts.get(s, 0) + 1
        return counts

    @staticmethod
    def parse_subset_ratio_spec(spec: str) -> Dict[str, float]:
        # format: "Sun360:1,ZInD:2,scene:0.5"
        ratio_map: Dict[str, float] = {}
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid subset ratio item '{item}', expected 'name:value'")
            name, value = item.split(":", 1)
            name = name.strip()
            value = value.strip()
            ratio = float(value)
            if ratio <= 0:
                raise ValueError(f"Subset ratio for '{name}' must be > 0, got {ratio}")
            ratio_map[name] = ratio
        if not ratio_map:
            raise ValueError("subset ratio spec is empty")
        return ratio_map

    def build_sample_weights(self, subset_ratio_spec: str) -> torch.Tensor:
        ratio_map = self.parse_subset_ratio_spec(subset_ratio_spec)
        known_subsets = set(self.subset_counts.keys())
        unknown_spec_keys = [k for k in ratio_map.keys() if k not in known_subsets]
        if unknown_spec_keys:
            print(f"[SubsetRatio] Warning: unknown subset keys in spec: {unknown_spec_keys}")

        # Normalize by subset size so aggregate sampling mass follows requested ratios.
        weights = []
        for subset in self.subsets:
            ratio = ratio_map.get(subset, 1.0)
            cnt = self.subset_counts[subset]
            weights.append(ratio / max(cnt, 1))
        return torch.tensor(weights, dtype=torch.double)

    def __len__(self) -> int:
        return len(self.ids)

    def _open_image(self, uri: str) -> Image.Image:
        if uri.startswith("s3://"):
            try:
                import boto3
            except ImportError as exc:
                raise ImportError("boto3 is required for s3 training data access") from exc

            if self._s3_client is None:
                self._s3_client = boto3.client("s3")
            bucket, key = _parse_s3_uri(uri)
            obj = self._s3_client.get_object(Bucket=bucket, Key=key)
            binary = obj["Body"].read()
            return Image.open(io.BytesIO(binary)).convert("RGB")

        return Image.open(uri).convert("RGB")

    def _build_uri(self, pano_id: str) -> str:
        return f"{self.pano_root}/{pano_id}.{self.image_ext}"

    def _sample_view_params(self) -> Tuple[torch.Tensor, int]:
        n_views = random.randint(self.min_views, self.max_views)
        params = torch.zeros((self.max_views, 3), dtype=torch.float32)
        for i in range(n_views):
            yaw = random.uniform(-180.0, 180.0)
            pitch = random.uniform(self.min_pitch, self.max_pitch)
            fov = random.uniform(self.min_fov, self.max_fov)
            params[i, 0] = yaw
            params[i, 1] = pitch
            params[i, 2] = fov
        return params, n_views

    def __getitem__(self, index: int) -> OutpaintSample:
        pano_id = self.ids[index]
        pano_uri = self._build_uri(pano_id)
        target_image = self._open_image(pano_uri)
        target_image = target_image.resize((self.pano_width, self.pano_height), Image.BICUBIC)
        target_tensor = self.target_transform(target_image)
        view_params, num_views = self._sample_view_params()
        caption = self._caption_map.get(pano_id, self.caption_template)

        return OutpaintSample(
            target_pixel_values=target_tensor,
            view_params=view_params,
            num_views=num_views,
            captions=caption,
        )


@torch.no_grad()
def build_condition_from_target(
    target_pixel_values: torch.Tensor,
    view_params: torch.Tensor,
    num_views: int,
    perspective_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # target_pixel_values: [3, H, W] in [-1, 1]
    device = target_pixel_values.device
    pano_01 = (target_pixel_values.float().clamp(-1.0, 1.0) * 0.5 + 0.5).unsqueeze(0)
    _, _, pano_h, pano_w = pano_01.shape

    canvas = torch.zeros((1, 3, pano_h, pano_w), dtype=torch.float32, device=device)
    known_mask = torch.zeros((1, 1, pano_h, pano_w), dtype=torch.float32, device=device)

    for i in range(int(num_views)):
        yaw = float(view_params[i, 0].item())
        pitch = float(view_params[i, 1].item())
        fov = float(view_params[i, 2].item())

        # Keep this convention as requested.
        proj_pitch = pitch + 180.0

        persp_rgb = pano_to_perspective(
            pano_rgb=pano_01,
            h_fov=fov,
            yaw=yaw,
            pitch=proj_pitch,
            persp_h=perspective_size,
            persp_w=perspective_size,
        )

        warped_rgb, warped_mask = perspective_to_pano_mask_only(
            persp_rgb=persp_rgb,
            h_fov=fov,
            yaw=yaw,
            pitch=proj_pitch,
            pano_h=pano_h,
            pano_w=pano_w,
        )
        valid_bool = warped_mask > 0.5
        canvas = torch.where(valid_bool.expand_as(canvas), warped_rgb, canvas)
        known_mask = torch.maximum(known_mask, warped_mask)

    condition = (canvas.clamp(0.0, 1.0) * 2.0 - 1.0).squeeze(0)
    unknown_mask = (known_mask < 0.5).float().squeeze(0)
    return condition, unknown_mask


def _norm_to_uint8_img(tensor: torch.Tensor) -> np.ndarray:
    # tensor: [3, H, W], normalized to [-1, 1]
    x = tensor.detach().cpu().clamp(-1.0, 1.0)
    x = (x * 0.5 + 0.5).clamp(0.0, 1.0)
    x = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return x


def _mask_to_uint8(mask: torch.Tensor) -> np.ndarray:
    # mask: [1, H, W], 1 means unknown(outpaint area)
    x = mask.detach().cpu().squeeze(0).numpy()
    x = (x > 0.5).astype(np.uint8) * 255
    return x


def _save_preview_html(
    out_dir: str,
    rows: List[Dict[str, str]],
    pano_root: str,
    id_list_file: str,
) -> str:
    html_path = os.path.join(out_dir, "index.html")
    parts = [
        "<html><head><meta charset='utf-8'><title>Outpaint Dataset Preview</title>",
        "<style>body{font-family:Arial,sans-serif;padding:20px;} table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #ccc;padding:8px;vertical-align:top;} img{max-width:100%;height:auto;}</style>",
        "</head><body>",
        "<h2>Outpaint Dataset Preview</h2>",
        f"<p><b>pano_root</b>: {pano_root}<br><b>id_list</b>: {id_list_file}</p>",
        "<table><tr><th>Sample</th><th>Target</th><th>Condition</th><th>Unknown Mask</th><th>Overlay</th><th>Caption</th></tr>",
    ]
    for row in rows:
        parts.append(
            "<tr>"
            f"<td>{row['sample_id']}</td>"
            f"<td><img src='{row['target']}'></td>"
            f"<td><img src='{row['condition']}'></td>"
            f"<td><img src='{row['mask']}'></td>"
            f"<td><img src='{row['overlay']}'></td>"
            f"<td>{row['caption']}</td>"
            "</tr>"
        )
    parts.append("</table></body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Quick preview for RandomPerspOutpaintDataset")
    parser.add_argument("--id_list_file", type=str, default="./data/panomtdu_train.json")
    parser.add_argument(
        "--pano_root",
        type=str,
        default="s3://adobe-lingzhi-p/jingdongz-data/PanoPseudoLabels_processed/img",
    )
    parser.add_argument("--output_dir", type=str, default="./dataset_preview_outpaint")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pano_height", type=int, default=512)
    parser.add_argument("--pano_width", type=int, default=1024)
    parser.add_argument("--image_ext", type=str, default="jpg")
    parser.add_argument("--min_views", type=int, default=1)
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--min_fov", type=float, default=75.0)
    parser.add_argument("--max_fov", type=float, default=105.0)
    parser.add_argument("--min_pitch", type=float, default=-30.0)
    parser.add_argument("--max_pitch", type=float, default=30.0)
    parser.add_argument("--perspective_size", type=int, default=512)
    parser.add_argument("--projection_device", type=str, default="auto", help="cpu | cuda | auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = RandomPerspOutpaintDataset(
        id_list_file=args.id_list_file,
        pano_root=args.pano_root,
        pano_height=args.pano_height,
        pano_width=args.pano_width,
        image_ext=args.image_ext,
        min_views=args.min_views,
        max_views=args.max_views,
        min_fov=args.min_fov,
        max_fov=args.max_fov,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
    )

    rows: List[Dict[str, str]] = []
    max_i = min(args.start_index + args.num_samples, len(dataset))
    if args.projection_device == "auto":
        proj_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        proj_device = torch.device(args.projection_device)

    for i in range(args.start_index, max_i):
        sample = dataset[i]
        sample_id = dataset.ids[i]

        target_np = _norm_to_uint8_img(sample.target_pixel_values)
        condition_tensor, unknown_mask = build_condition_from_target(
            target_pixel_values=sample.target_pixel_values.to(proj_device),
            view_params=sample.view_params.to(proj_device),
            num_views=sample.num_views,
            perspective_size=args.perspective_size,
        )
        condition_np = _norm_to_uint8_img(condition_tensor.cpu())
        mask_np = _mask_to_uint8(unknown_mask.cpu())

        overlay_np = target_np.copy()
        # Highlight unknown area in red for easier sanity check.
        overlay_np[mask_np > 0] = (0.65 * overlay_np[mask_np > 0] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)

        target_name = f"{i:06d}_{sample_id}_target.png"
        cond_name = f"{i:06d}_{sample_id}_condition.png"
        mask_name = f"{i:06d}_{sample_id}_unknown_mask.png"
        overlay_name = f"{i:06d}_{sample_id}_overlay.png"

        Image.fromarray(target_np).save(os.path.join(args.output_dir, target_name))
        Image.fromarray(condition_np).save(os.path.join(args.output_dir, cond_name))
        Image.fromarray(mask_np).save(os.path.join(args.output_dir, mask_name))
        Image.fromarray(overlay_np).save(os.path.join(args.output_dir, overlay_name))

        rows.append(
            {
                "sample_id": sample_id,
                "target": target_name,
                "condition": cond_name,
                "mask": mask_name,
                "overlay": overlay_name,
                "caption": sample.captions,
            }
        )

    html_path = _save_preview_html(
        out_dir=args.output_dir,
        rows=rows,
        pano_root=args.pano_root,
        id_list_file=args.id_list_file,
    )
    print(f"Saved preview to: {html_path}")


if __name__ == "__main__":
    main()
