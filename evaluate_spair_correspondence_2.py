from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import hydra
import numpy as np
import torch
import torch.nn.functional as nn_F
from einops import einsum
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from PIL import Image

from evals.datasets.spair import CLASS_IDS, SPairDataset
from evals.utils.correspondence import argmax_2d

from hydra.core.hydra_config import HydraConfig
import os

# ========== Helper ==========

def to_numpy(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)

def visualize_matching_gif(img_a, img_b, kps_a, kps_b, pred_kps, heatmaps, save_path="matching.gif"):
    img_a_np = to_numpy(img_a)
    img_b_np = to_numpy(img_b)

    H, W = img_a_np.shape[:2]
    frames = []

    for idx in range(len(kps_a)):
        pt_a = (int(kps_a[idx, 0]), int(kps_a[idx, 1]))
        pt_b_gt = (int(kps_b[idx, 0]), int(kps_b[idx, 1]))
        pt_b_pred = (int(pred_kps[idx, 0] * W), int(pred_kps[idx, 1] * H))

        img_a_frame = img_a_np.copy()
        img_b_frame = img_b_np.copy()

        cv2.circle(img_a_frame, pt_a, 5, (1, 0, 0), -1)         # Red in Image A
        cv2.circle(img_b_frame, pt_b_gt, 5, (0, 1, 0), -1)      # Green GT in Image B
        cv2.circle(img_b_frame, pt_b_pred, 5, (1, 0, 0), -1)    # Red pred in Image B

        # Add heatmap overlay
        heatmap = heatmaps[idx].detach().cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_colored = (plt.cm.jet(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        heatmap_overlay = (0.3 * heatmap_colored + 0.7 * img_b_frame)

        # Stack image A and overlay side-by-side
        combined = np.concatenate([img_a_frame, heatmap_overlay], axis=1)
        combined = (combined * 255).astype(np.uint8) if combined.max() <= 1.0 else combined.astype(np.uint8)
        frame = Image.fromarray(combined)
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=600,
        loop=0
        )
    print(f"GIF saved to {save_path}")
    
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
thickness = 2

def visualize_matching(img_a, img_b, kps_a, kps_b, pred_kps, heatmaps, save_path=None):
    img_a_np = to_numpy(img_a)
    img_b_np = to_numpy(img_b)

    H, W = img_a_np.shape[:2]
    img_a_draw = (img_a_np.copy() * 255).astype(np.uint8)
    img_b_draw = (img_b_np.copy() * 255).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for idx in range(len(kps_a)):
        pt_a = (int(kps_a[idx, 0]), int(kps_a[idx, 1]))
        pt_b_gt = (int(kps_b[idx, 0]), int(kps_b[idx, 1]))
        pt_b_pred = (int(pred_kps[idx, 0] * W), int(pred_kps[idx, 1] * H))

        text = str(idx)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        offset = (text_width // 2, text_height // 2)

        cv2.putText(img_a_draw, text, (pt_a[0] - offset[0], pt_a[1] + offset[1]), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(img_b_draw, text, (pt_b_gt[0] - offset[0], pt_b_gt[1] + offset[1]), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(img_b_draw, text, (pt_b_pred[0] - offset[0], pt_b_pred[1] + offset[1]), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

    # Add white spacing between the two images
    spacing = 20
    canvas_height = max(img_a_draw.shape[0], img_b_draw.shape[0])
    canvas_width = img_a_draw.shape[1] + img_b_draw.shape[1] + spacing
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Paste the images
    canvas[:img_a_draw.shape[0], :img_a_draw.shape[1]] = img_a_draw
    canvas[:img_b_draw.shape[0], img_a_draw.shape[1] + spacing:] = img_b_draw

    # Convert BGR to RGB and normalize
    # canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas_rgb = canvas.astype(np.float32) / 255.0

    plt.figure(figsize=(14, 7))
    plt.imshow(canvas_rgb)
    plt.title("Keypoint Matching: Image A (Left) and Image B (Right) with GT (Green), Pred (Red)")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()
        
def visualize_matching_old(img_a, img_b, kps_a, kps_b, pred_kps, heatmaps, save_path=None):
    img_a_np = to_numpy(img_a)
    img_b_np = to_numpy(img_b)

    H, W = img_a_np.shape[:2]
    img_a_draw = img_a_np.copy()
    img_b_draw = img_b_np.copy()

    # for idx in range(len(kps_a)):
    #     pt_a = (kps_a[idx, 0], kps_a[idx, 1])
    #     pt_b_gt = (kps_b[idx, 0], kps_b[idx, 1])
    #     pt_b_pred = (pred_kps[idx, 0] * W, pred_kps[idx, 1] * H)

    #     cv2.circle(img_a_draw, (int(pt_a[0]), int(pt_a[1])), 3, (1, 0, 0), -1)
    #     cv2.circle(img_b_draw, (int(pt_b_gt[0]), int(pt_b_gt[1])), 3, (0, 1, 0), -1)
    #     cv2.circle(img_b_draw, (int(pt_b_pred[0]), int(pt_b_pred[1])), 3, (1, 0, 0), -1)
    for idx in range(len(kps_a)):
        pt_a = (int(kps_a[idx, 0]), int(kps_a[idx, 1]))
        pt_b_gt = (int(kps_b[idx, 0]), int(kps_b[idx, 1]))
        pt_b_pred = (int(pred_kps[idx, 0] * W), int(pred_kps[idx, 1] * H))

        text = str(idx)

        # Get text size to center the index on keypoint
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        offset = (text_width // 2, text_height // 2)

        # Draw centered index at each keypoint
        cv2.putText(img_a_draw, text, (pt_a[0] - offset[0], pt_a[1] + offset[1]), font, font_scale, (1, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(img_b_draw, text, (pt_b_gt[0] - offset[0], pt_b_gt[1] + offset[1]), font, font_scale, (0, 1, 0), thickness, cv2.LINE_AA)
        cv2.putText(img_b_draw, text, (pt_b_pred[0] - offset[0], pt_b_pred[1] + offset[1]), font, font_scale, (1, 0, 0), thickness, cv2.LINE_AA)

    # Visualize all heatmaps in a grid
    num_keypoints = heatmaps.shape[0]
    grid_cols = min(8, num_keypoints)
    grid_rows = int(np.ceil(num_keypoints / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))
    for idx in range(num_keypoints):
        heatmap = heatmaps[idx].detach().cpu().numpy()
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_overlay = (0.3 * heatmap_colored + 0.7 * img_b_np)
        r, c = divmod(idx, grid_cols)
        axes[r, c].imshow(heatmap_overlay)
        axes[r, c].set_title(f"KP {idx}")
        axes[r, c].axis('off')
    for idx in range(num_keypoints, grid_rows * grid_cols):
        r, c = divmod(idx, grid_cols)
        axes[r, c].axis('off')

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
    axes2[0].imshow(img_a_draw)
    axes2[0].set_title("Image A with All Keypoints")
    axes2[0].axis('off')
    axes2[1].imshow(img_b_draw)
    axes2[1].set_title("Image B with GT (Green) and Pred (Red)")
    axes2[1].axis('off')
    fig2.tight_layout()
    fig.tight_layout()
    fig2.tight_layout()
    if save_path:
        combined_fig, combined_axes = plt.subplots(2, 1, figsize=(max(fig.get_size_inches()[0], fig2.get_size_inches()[0]), fig.get_size_inches()[1] + fig2.get_size_inches()[1]))

        fig.canvas.draw()
        fig2.canvas.draw()

        heatmap_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
        kp_img = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig2.canvas.get_width_height()[::-1] + (3,))

        combined_axes[0].imshow(kp_img)
        combined_axes[0].axis('off')
        combined_axes[1].imshow(heatmap_img)
        combined_axes[1].axis('off')
        combined_fig.tight_layout()
        combined_fig.savefig(save_path, dpi=300)
        plt.close('all')
        plt.close()
    else:
        plt.show()

def compute_errors(model, instance, mask_feats=False, return_heatmaps=False):
    img_i, mask_i, kps_i, img_j, mask_j, kps_j, thresh_scale, _ = instance
    mask_i = torch.tensor(np.array(mask_i, dtype=float))
    mask_j = torch.tensor(np.array(mask_j, dtype=float))

    images = torch.stack((img_i, img_j)).cuda()
    masks = torch.stack((mask_i, mask_j)).cuda()
    masks = torch.nn.functional.avg_pool2d(masks.float(), 16)
    masks = masks > 4 / (16 ** 2)

    feats = model(images)
    if isinstance(feats, list):
        feats = torch.cat(feats, dim=1)

    feats = nn_F.normalize(feats, p=2, dim=1)

    if mask_feats:
        feats = feats * masks

    feats_i = feats[0]
    feats_j = feats[1]

    # normalize kps to [0, 1]
    assert images.shape[-1] == images.shape[-2], "assuming square images here"
    kps_i = kps_i.float()
    kps_j = kps_j.float()
    kps_i[:, :2] = kps_i[:, :2] / images.shape[-1]
    kps_j[:, :2] = kps_j[:, :2] / images.shape[-1]

    # get correspondences
    kps_i_ndc = (kps_i[:, :2].float() * 2 - 1)[None, None].cuda()
    kp_i_F = nn_F.grid_sample(
        feats_i[None, :], kps_i_ndc, mode="bilinear", align_corners=True
    )
    kp_i_F = kp_i_F[0, :, 0].t()

    # get max index in [0,1] range
    heatmaps = einsum(kp_i_F, feats_j, "k f, f h w -> k h w")
    pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / feats.shape[-1]

    # compute error and scale to threshold (for all pairs)
    errors = (pred_kp[:, None, :] - kps_j[None, :, :2]).norm(p=2, dim=-1)
    errors = errors / thresh_scale

    # only retain keypoints in both (for now)
    valid_kps = (kps_i[:, None, 2] * kps_j[None, :, 2]) == 1
    in_both = valid_kps.diagonal()

    # max error should be 1, so this excludes invalid from NN-search
    errors[valid_kps.logical_not()] = 1e3

    error_same = errors.diagonal()[in_both]
    error_nn, index_nn = errors[in_both].min(dim=1)
    index_same = in_both.nonzero().squeeze(1)

    if return_heatmaps:
        return error_same, error_nn, index_same, index_nn, heatmaps
    else:
        return error_same, error_nn, index_same, index_nn


def evaluate_dataset(model, dataset, thresh, verbose=False):
    pbar = tqdm(range(len(dataset)), ncols=60) if verbose else range(len(dataset))
    error_output = [compute_errors(model, dataset.__getitem__(i)) for i in pbar]

    errors = torch.cat([_err[0] for _err in error_output])
    src_ind = torch.cat([_err[2] for _err in error_output])
    tgt_ind = torch.cat([_err[3] for _err in error_output])

    # compute confusion matrix
    kp_max = max(src_ind.max(), tgt_ind.max()) + 1
    confusion = torch.zeros((kp_max, kp_max))
    for src, tgt in torch.stack((src_ind, tgt_ind), dim=1):
        confusion[src, tgt] += 1

    # compute recall
    recall = (errors < thresh).float().mean().item() * 100.0

    return recall, confusion

@hydra.main(config_path="configs", config_name="spair_correspondence", version_base=None)
def main(cfg: DictConfig):
    output_dir = HydraConfig.get().run.dir
    print(f'Output dir: {output_dir}')
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    data_root = "data/SPair-71k"
    thresh = 0.10

    model = instantiate(cfg.backbone, output="dense", return_multilayer=cfg.multilayer)
    model = model.to("cuda")

    if cfg.eval_class == "all":
        classes = list(CLASS_IDS.keys())
    else:
        assert cfg.eval_class in CLASS_IDS
        classes = [cfg.eval_class]

    for class_name in classes:
        for vp_diff in [0, 1, 2, None]:
            dataset = SPairDataset(
                data_root,
                cfg.split,
                use_bbox=cfg.use_bbox,
                image_size=cfg.image_size,
                image_mean=cfg.image_mean,
                class_name=class_name,
                num_instances=cfg.num_instances,
                vp_diff=vp_diff,
            )
            vp_diff_str = "all" if vp_diff is None else f"{vp_diff:3d}"
            if len(dataset) == 0:
                continue

            for i in range(len(dataset)):
                if i % 100 != 0:
                    continue
                instance = dataset[i]
                error_same, error_nn, index_same, index_nn, heatmaps = compute_errors(
                    model, instance, mask_feats=False, return_heatmaps=True
                )

                img_i, mask_i, kps_i, img_j, mask_j, kps_j, _, _ = instance
                img_i = img_i.cuda()
                img_j = img_j.cuda()
                kps_i_norm = kps_i[:, :2].cpu().numpy()
                kps_j_norm = kps_j[:, :2].cpu().numpy()
                pred_kp = argmax_2d(heatmaps, max_value=True).float().cpu() / heatmaps.shape[-1]
                pred_kp_np = pred_kp.numpy()

                vis_path = os.path.join(vis_dir, f"{class_name}_{vp_diff_str}_sample{i:04d}.png")
                # vis_path = os.path.join(vis_dir, f"{class_name}_{vp_diff_str}_sample{i:04d}.gif")
                visualize_matching(
                    img_i, img_j, kps_i_norm, kps_j_norm, pred_kp_np, heatmaps, save_path=vis_path
                )

if __name__ == "__main__":
    main()
