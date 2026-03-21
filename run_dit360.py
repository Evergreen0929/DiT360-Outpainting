import argparse
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.pipeline import DiT360Pipeline
from diffusers.utils import load_image
import torchvision

from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor

# ==========================================
# Part 1: Geometry Utils (Adapted from your panorama_utils.py)
# ==========================================

def get_camera_matrices(yaw, pitch, device='cpu'):
    """
    构建相机到世界的旋转矩阵和世界到相机的旋转矩阵。
    Adapted from panorama_utils.py
    """
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)

    # 围绕世界Y轴的偏航旋转
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    R_yaw = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ], device=device, dtype=torch.float32)

    # 围绕相机局部X轴的俯仰旋转
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)
    R_pitch = torch.tensor([
        [1, 0, 0],
        [0, cos_p, -sin_p],
        [0, sin_p, cos_p]
    ], device=device, dtype=torch.float32)

    # 组合旋转: 首先应用俯仰，然后应用偏航
    cam_to_world_rot = R_yaw @ R_pitch
    world_to_cam_rot = cam_to_world_rot.T

    return cam_to_world_rot, world_to_cam_rot

def perspective_to_pano_mask_only(persp_rgb, h_fov, yaw, pitch, pano_h, pano_w):
    """
    仅用于 Outpaint 预处理：将透视 RGB 投影到全景画布，并生成有效区域 Mask。
    Simplified from panorama_utils.py (perspective_to_pano_correct)
    """
    device = persp_rgb.device
    B, _, persp_h, persp_w = persp_rgb.shape
    
    cam_to_world_rot, world_to_cam_rot = get_camera_matrices(yaw, pitch, device)

    # 1. 为全景图的每个像素计算世界空间中的射线 (N-Space)
    # Your code uses 'ij' indexing
    v, u = torch.meshgrid(
        torch.linspace(-1, 1, pano_h, device=device),
        torch.linspace(-1, 1, pano_w, device=device),
        indexing='ij'
    )
    theta = u * math.pi
    phi = v * math.pi / 2
    
    # N-Space rays (World space) based on your utils logic:
    # S-Space: s_x, s_y, s_z derived first, then N-Space is [s_z, s_y, s_x]
    s_x = torch.cos(phi) * torch.sin(theta)
    s_y = torch.sin(phi)
    s_z = -torch.cos(phi) * torch.cos(theta)
    rays_world_N_space = torch.stack([s_z, s_y, s_x], dim=-1) # (H, W, 3)

    # 2. 将世界射线变换到相机空间
    # (H, W, 3) @ (3, 3).T
    rays_cam = rays_world_N_space.view(-1, 3) @ world_to_cam_rot.T
    rays_cam = rays_cam.view(pano_h, pano_w, 3)

    # 3. 将相机射线投影到透视图像平面以创建采样网格
    cx, cy, cz = rays_cam.unbind(dim=-1)
    
    # 掩码：只处理指向相机前方(-Z)的射线
    mask_cam_front = (cz < -1e-8)
    
    h_fov_rad = math.radians(h_fov)
    # Focal length calculation
    focal_length = persp_w / (2 * math.tan(h_fov_rad / 2))
    
    # Project to image plane
    u_proj = focal_length * (cx / -cz)
    v_proj = focal_length * (cy / -cz)

    # Normalize to [-1, 1] for grid_sample
    u_norm = u_proj / (persp_w / 2)
    v_norm = v_proj / (persp_h / 2)
    
    # Stack grid: (H, W, 2)
    # Your code uses [u_norm, -v_norm]
    grid = torch.stack([u_norm, -v_norm], dim=-1)

    # Valid mask in grid space
    valid_sample_mask = mask_cam_front & (u_norm.abs() <= 1) & (v_norm.abs() <= 1)
    
    # Mask out invalid regions in grid to avoid sampling garbage
    grid[~valid_sample_mask] = 2.0 

    grid_b = grid.unsqueeze(0).expand(B, -1, -1, -1) # (B, H, W, 2)
    
    # 4. grid_sample
    warped_rgb = F.grid_sample(persp_rgb, grid_b, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # Create final mask (1 for valid image area, 0 for empty area)
    valid_mask_b = valid_sample_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1).float()
    
    return warped_rgb, valid_mask_b

# ==========================================
# Part 2: Main Logic
# ==========================================

def run_text_to_pano(args, pipe, device):
    print(f"--- Mode: Text-to-Pano ---")
    print(f"Prompt: {args.prompt}")
    
    image = pipe(
        prompt=args.prompt,
        width=2048,
        height=1024,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        generator=torch.Generator(device=device).manual_seed(args.seed)
    ).images[0]
    
    image.save(args.output)
    print(f"Saved result to: {args.output}")

def run_outpaint(args, pipe, device):
    print(f"--- Mode: Outpaint ---")
    if not args.image:
        raise ValueError("Outpaint mode requires --image input.")

    # =========================================================================
    # 1. 重新加载专用 Pipeline
    #    (T2I 用的是 DiT360Pipeline，但 Outpaint 必须用 pa_src 里的 RFPanoInversionParallelFluxPipeline)
    # =========================================================================
    print("Switching pipeline for Outpaint (RFPanoInversionParallelFluxPipeline)...")
    del pipe # 释放旧模型的显存
    torch.cuda.empty_cache()
    
    # 重新加载模型
    # 注意：这里我们假设你的权重路径还是 "Insta360-Research/DiT360-Panorama-Image-Generation"
    pipe = RFPanoInversionParallelFluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")
    
    # =========================================================================
    # 2. 准备数据 (Warp 图片)
    # =========================================================================
    print(f"Loading input image: {args.image}")
    pil_img = Image.open(args.image).convert("RGB")
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"Projecting image (FOV: {args.fov})...")
    # 使用之前的投影函数得到全景画布和 mask (1=Keep, 0=Gen)
    warped_rgb, mask_tensor = perspective_to_pano_mask_only(
        input_tensor, 
        h_fov=args.fov, 
        yaw=args.yaw, 
        pitch=args.pitch + 180, 
        pano_h=1024, 
        pano_w=2048
    )
    
    # 转回 PIL 供 pipeline 使用
    init_image = torchvision.transforms.ToPILImage()(warped_rgb.squeeze(0).cpu())
    
    # 保存调试图
    debug_dir = os.path.dirname(args.output) or "."
    init_image.save(os.path.join(debug_dir, "debug_warped_canvas.png"))
    torchvision.transforms.ToPILImage()(mask_tensor.squeeze(0).cpu()).save(os.path.join(debug_dir, "debug_mask.png"))
    
    # =========================================================================
    # 3. 核心逻辑 (完全复刻 editing.py)
    # =========================================================================
    height = 1024
    width = 2048
    timestep = args.steps # 默认 50
    tau = 50  # 强度控制 (0-100)，越小一致性越强
    
    # 计算 Latent 尺寸 (参考 editing.py)
    # pipe.vae_scale_factor 通常是 8，这里除以 16
    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * (latent_w + 2) # +2 是因为下面做了 padding

    # 处理 Mask：下采样到 Latent 尺寸
    # mask_tensor shape is (1, 1, 1024, 2048). We need (latent_h, latent_w)
    mask_resized = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode='nearest')
    mask_resized = mask_resized.squeeze(0).squeeze(0).float().to(device) # (H, W)
    
    # editing.py 中的特殊 padding 逻辑:
    # "mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)"
    # 这一步是为了处理全景图左右边缘的循环一致性 (circular padding)
    mask_processed = torch.cat([mask_resized[:, 0:1], mask_resized, mask_resized[:, -1:]], dim=-1).view(-1, 1)

    print("Running Inversion...")
    inverted_latents, image_latents, latent_image_ids = pipe.invert( 
        source_prompt="", 
        image=init_image, 
        height=height,
        width=width,
        num_inversion_steps=timestep, 
        gamma=1.0
    )

    print("Setting up Attention Processor...")
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau/100, mask=mask_processed, device=device, img_dims=img_dims),
    )

    print("Running Generation...")
    # prompt 和 new_prompt 通常保持一致即可
    image = pipe(
        [args.prompt, args.prompt], 
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        start_timestep=0.0, 
        stop_timestep=0.99,
        num_inference_steps=timestep,
        eta=1.0, 
        generator=torch.Generator(device=device).manual_seed(args.seed),
        mask=mask_processed,
        use_timestep=True
    ).images[1] # images[1] 是 new_prompt 的结果

    image.save(args.output)
    print(f"Outpaint result saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="DiT360 Unified Script")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode: t2i or outpaint")

    # Common args
    parser.add_argument("--output", type=str, default="result.png", help="Output filename")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    
    # Text to Pano args
    parser_t2i = subparsers.add_parser('t2i', help="Text to Panorama")
    parser_t2i.add_argument("--prompt", type=str, required=True)

    # Outpaint args
    parser_out = subparsers.add_parser('outpaint', help="Outpaint from Perspective Image")
    parser_out.add_argument("--prompt", type=str, required=True)
    parser_out.add_argument("--image", type=str, required=True, help="Input perspective image path")
    parser_out.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV of input image")
    parser_out.add_argument("--yaw", type=float, default=-90.0, help="Yaw angle to place image (Degrees)")
    parser_out.add_argument("--pitch", type=float, default=0.0, help="Pitch angle to place image (Degrees)")

    args = parser.parse_args()

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DiT360 Pipeline on {device}...")
    
    # Ensure you are logged into huggingface-cli or have access
    pipe = DiT360Pipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16
    ).to(device)
    pipe.load_lora_weights("Insta360-Research/DiT360-Panorama-Image-Generation")

    if args.mode == "t2i":
        run_text_to_pano(args, pipe, device)
    elif args.mode == "outpaint":
        run_outpaint(args, pipe, device)

if __name__ == "__main__":
    main()