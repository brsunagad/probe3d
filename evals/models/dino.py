import os
import re

import torch
import torch.nn as nn

from .utils import center_padding, tokens_to_output


class DINO(torch.nn.Module):
    def __init__(
        self,
        dino_name="dino",
        model_name="vitb16",
        output="dense",
        layer=-1,
        return_multilayer=False,
        teacher_checkpoint=None,
    ):
        super().__init__()
        feat_dims = {
            "vitb8": 768,
            "vitb16": 768,
            "vitb14": 768,
            "vitb14_reg": 768,
            "vitl14": 1024,
            "vitg14": 1536,
        }

        # get model
        self.model_name = dino_name
        self.checkpoint_name = f"{dino_name}_{model_name}"

        if teacher_checkpoint is not None:
            # Extract checkpoint info
            path_parts = teacher_checkpoint.split("/")
            output_info = next(
                (part for part in path_parts if part.startswith("outputs_")), ""
            )
            match = re.search(r"training_(\d+)", teacher_checkpoint)
            iter_number = match.group(1) if match else "unknown"

            self.checkpoint_name += f"-{output_info}-{iter_number}-teacher"

        self.vit = DINO.create_model(
            model_type=dino_name,
            arch=model_name,
            pretrained_teacher_path=teacher_checkpoint if teacher_checkpoint else None,
        )

        self.has_registers = "_reg" in model_name

        assert output in ["cls", "gap", "dense", "dense-cls"]
        self.output = output
        self.patch_size = self.vit.patch_embed.proj.kernel_size[0]

        feat_dim = feat_dims[model_name]
        feat_dim = feat_dim * 2 if output == "dense-cls" else feat_dim

        num_layers = len(self.vit.blocks)

        print(self.vit.blocks)
        multilayers = [
            num_layers // 4 - 1,
            num_layers // 2 - 1,
            num_layers // 4 * 3 - 1,
            num_layers - 1,
        ]

        if return_multilayer:
            self.feat_dim = [feat_dim, feat_dim, feat_dim, feat_dim]
            self.multilayers = multilayers
        else:
            self.feat_dim = feat_dim
            layer = multilayers[-1] if layer == -1 else layer
            self.multilayers = [layer]

        # define layer name (for logging)
        self.layer = "-".join(str(_x) for _x in self.multilayers)

    @staticmethod
    def create_model(
        model_type: str, arch: str, pretrained_teacher_path: str = None
    ) -> nn.Module:
        """
        Args:
            model_type (str): Type of ViT model (e.g., "dino", "dinov2", "vanilla").
            arch (str): Architecture of ViT (e.g., "vitb16", "vitl14").
            pretrained_teacher_path (str, optional): Path to a saved teacher checkpoint.
        Returns:
            model (nn.Module): Initialized model.
        """
        from .custom_dino_utils.vision_transformer import (
            vit_base,  # Assuming you have the ViT code from DINOv2.
        )

        if pretrained_teacher_path is not None:
            # 1. Build a fresh model (same architecture) with random weights
            if arch == "vitb14" or arch == "vitb14_reg":
                model = vit_base(
                    patch_size=14,
                    block_chunks=4,
                    img_size=224,
                    num_register_tokens=4 if "reg" in arch else 0,
                    interpolate_antialias=True,
                    ffn_layer="swiglufused",  # must match training config
                    drop_path_rate=0.2,
                    init_values=1e-6,
                )
            else:
                raise ValueError(f"Unsupported arch for DINOv2 loading: {arch}")

            # 2. Load teacher weights
            print(f"Loading teacher checkpoint from {pretrained_teacher_path}")
            checkpoint = torch.load(pretrained_teacher_path, map_location="cpu")
            teacher_state_dict = checkpoint["teacher"]
            # teacher_state_dict = DINO.flatten_checkpoint(teacher_state_dict)

            # 🧹 Filter and rename keys
            new_teacher_state_dict = {}
            for k, v in teacher_state_dict.items():
                if k.startswith("backbone."):
                    new_key = k[len("backbone.") :]  # remove "backbone." prefix
                    new_teacher_state_dict[new_key] = v
                else:
                    # Ignore keys like "dino_head.*"
                    pass
            missing, unexpected = model.load_state_dict(
                new_teacher_state_dict, strict=False
            )
            model = DINO.flatten_blockchunks(model)

            if missing:
                print("⚠️ Missing keys:")
                for key in missing:
                    print(f"   {key}")
            if unexpected:
                print("⚠️ Unexpected keys:")
                for key in unexpected:
                    print(f"   {key}")

        else:
            # fallback to hub/timm
            if "dino" in model_type:
                model = torch.hub.load(
                    f"facebookresearch/{model_type}:main", f"{model_type}_{arch}"
                )

        return model.eval().to(torch.float32)

    @staticmethod
    def flatten_blockchunks(model):
        """
        Flattens BlockChunks into a simple flat ModuleList of NestedTensorBlocks,
        ignoring Identity layers.
        """
        if not hasattr(model, "blocks"):
            raise ValueError("Model has no 'blocks' attribute.")

        all_blocks = []
        for chunk in model.blocks:
            for block in chunk.children():
                if isinstance(block, torch.nn.Identity):
                    continue  # 🚫 Ignore identity blocks
                all_blocks.append(block)

        model.blocks = torch.nn.ModuleList(all_blocks)
        print(f"✅ Flattened to {len(model.blocks)} blocks (identities ignored).")

        return model

    @staticmethod
    def flatten_checkpoint(teacher_ckpt):
        """Fix BlockChunk nested keys for loading into flat block models with skipped identity blocks."""

        # This is based on your model structure:
        # - Block 0: blocks 0-2 are real
        # - Block 1: blocks 3-5 real, 0-2 are Identity (so need to shift)
        # - Block 2: blocks 6-8 real, 0-5 are Identity
        # - Block 3: blocks 9-11 real, 0-8 are Identity

        new_ckpt = {}
        block_shift = 0  # the new flattened block index

        for k, v in teacher_ckpt.items():
            if not k.startswith("blocks."):
                # keep non-block keys (like patch_embed, cls_token, etc.)
                new_ckpt[k] = v
                continue

            parts = k.split(".")
            chunk_idx = int(parts[1])
            inner_idx = int(parts[2])

            # In your checkpoint:
            # - in chunk 0, blocks 0-2 real
            # - in chunk 1, blocks 3-5 real
            # - in chunk 2, blocks 6-8 real
            # - in chunk 3, blocks 9-11 real

            if chunk_idx == 0 and inner_idx in [0, 1, 2]:
                flat_idx = inner_idx
            elif chunk_idx == 1 and inner_idx in [3, 4, 5]:
                flat_idx = inner_idx
            elif chunk_idx == 2 and inner_idx in [6, 7, 8]:
                flat_idx = inner_idx
            elif chunk_idx == 3 and inner_idx in [9, 10, 11]:
                flat_idx = inner_idx
            else:
                # this is identity block (was not learned), so skip it
                continue

            new_key = f"blocks.{flat_idx}." + ".".join(parts[3:])
            new_ckpt[new_key] = v

        return new_ckpt

    def forward(self, images):

        # pad images (if needed) to ensure it matches patch_size
        images = center_padding(images, self.patch_size)
        h, w = images.shape[-2:]
        h, w = h // self.patch_size, w // self.patch_size

        if self.model_name == "dinov2":
            x = self.vit.prepare_tokens_with_masks(images, None)
        else:
            x = self.vit.prepare_tokens(images)

        embeds = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in self.multilayers:
                embeds.append(x)
                if len(embeds) == len(self.multilayers):
                    break

        num_spatial = h * w
        outputs = []
        for i, x_i in enumerate(embeds):
            cls_tok = x_i[:, 0]
            # ignoring register tokens
            spatial = x_i[:, -1 * num_spatial :]
            x_i = tokens_to_output(self.output, spatial, cls_tok, (h, w))
            outputs.append(x_i)

        return outputs[0] if len(outputs) == 1 else outputs
