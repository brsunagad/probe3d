import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from projection_network import AggregationNetwork, DummyAggregationNetwork
from utils import normalize_feats


def demo_dino():

    print("Postprocessing DINO features")

    device = "cuda:0"
    num_patches = 60

    # Dummy network
    # aggre_net = DummyAggregationNetwork()

    # DINOv2
    ckpt_dir = "ckpts/dino_spair_0300.pth"
    feature_dims = [768]
    aggre_net = AggregationNetwork(
        feature_dims=feature_dims,
        projection_dim=768,
        device=device,
        feat_map_dropout=0.2,
    )
    pretrained_dict = torch.load(ckpt_dir)
    aggre_net.load_pretrained_weights(pretrained_dict)
    aggre_net.eval()

    # Load features
    desc_dino = torch.load("data/2008_000187_dino.pt", map_location=device)
    print(f"Loaded DINO features of shape {desc_dino.shape}")
    if desc_dino.shape[-2:] != (num_patches, num_patches):
        desc_dino = F.interpolate(
            desc_dino,
            size=(num_patches, num_patches),
            mode="bilinear",
            align_corners=False,
        )

    # Project features
    with torch.no_grad():
        desc_proj = aggre_net(desc_dino)
    print(f"Projected features of shape {desc_proj.shape}")

    # -------------------------
    # Example feature similarities of two images
    desc_dino_2 = torch.load("data/2008_006885_dino.pt", map_location=device)
    with torch.no_grad():
        desc_proj_2 = aggre_net(desc_dino_2)

    # Normalize features
    desc_proj = desc_proj.reshape(1, -1, num_patches ** 2).permute(0, 2, 1)
    desc_proj_2 = desc_proj_2.reshape(1, -1, num_patches ** 2).permute(0, 2, 1)
    desc_proj = normalize_feats(desc_proj)[0]
    desc_proj_2 = normalize_feats(desc_proj_2)[0]

    # Compute cosine similarity
    sim_ij = torch.matmul(desc_proj, desc_proj_2.permute(1, 0)).cpu().numpy()
    example_indx = 30 * num_patches + 30
    sim_ij = sim_ij[example_indx].reshape(num_patches, num_patches)

    plt.imshow(sim_ij)
    plt.colorbar()
    plt.savefig("viz/dino_similarity.png")


def demo_sd_dino():

    print("Postprocessing SD+DINO features")

    device = "cuda:0"
    num_patches = 60

    # DINOv2
    ckpt_dir = "ckpts/sd_dino_spair_0280.pth"
    feature_dims = [640, 1280, 1280, 768]
    aggre_net = AggregationNetwork(
        feature_dims=feature_dims,
        projection_dim=768,
        device=device,
        feat_map_dropout=0.2,
    )
    pretrained_dict = torch.load(ckpt_dir)
    aggre_net.load_pretrained_weights(pretrained_dict)
    aggre_net.eval()

    # Load features
    desc_dino = torch.load("data/2008_000187_dino.pt", map_location=device)
    desc_sd = torch.load("data/2008_000187_sd.pt", map_location=device)

    for k in desc_sd:
        desc_sd[k] = desc_sd[k].to(device)
    desc_gathered = torch.cat(
        [
            desc_sd["s3"],
            F.interpolate(
                desc_sd["s4"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            F.interpolate(
                desc_sd["s5"],
                size=(num_patches, num_patches),
                mode="bilinear",
                align_corners=False,
            ),
            desc_dino,
        ],
        dim=1,
    )
    print(f"Loaded SD+DINO features of shape {desc_gathered.shape}")

    # Project features
    with torch.no_grad():
        desc_proj = aggre_net(desc_gathered)
    print(f"Projected features of shape {desc_proj.shape}")


if __name__ == "__main__":
    demo_dino()
    demo_sd_dino()
