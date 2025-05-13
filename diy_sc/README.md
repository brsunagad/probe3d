# DIY-SC

This repo contains an example use of the leight-weight adapter that projects DINOv2 or SD+DINOv2 features to features that are better suited for finding semantic correspondences.

For the training of the adapter, SD and DINOv2 features were computed using the pipeline in https://github.com/Junyi42/GeoAware-SC/blob/master/preprocess_map.py.


## Requirements
- torch
- numpy
- matplotlib


## Example usage

DINOv2
```
# Load model
ckpt_dir = 'ckpts/dino_spair_0300.pth'
feature_dims = [768,]
aggre_net = AggregationNetwork(feature_dims=feature_dims, projection_dim=768, device=device, feat_map_dropout=0.2)
pretrained_dict = torch.load(ckpt_dir)
aggre_net.load_pretrained_weights(pretrained_dict)

# Project features
desc_dino = <DINOv2 features of torch.Size([1, 768, 60, 60])>
with torch.no_grad():
    desc_proj = aggre_net(desc_dino) # of shape torch.Size([1, 768, 60, 60])
```

SD+DINOv2
```
# Load model
ckpt_dir = 'ckpts/sd_dino_spair_0280.pth'
feature_dims = [640,1280,1280,768]
aggre_net = AggregationNetwork(feature_dims=feature_dims, projection_dim=768, device=device, feat_map_dropout=0.2)
pretrained_dict = torch.load(ckpt_dir)
aggre_net.load_pretrained_weights(pretrained_dict)

# Project features
desc_dino = <SD+DINOv2 features of torch.Size([1, 3968, 60, 60])>
with torch.no_grad():
    desc_proj = aggre_net(desc_dino) # of shape torch.Size([1, 768, 60, 60])
```

The script `example.py` runs two functions, illustrating how to project DINOv2 / SD+DINOv2 features.
Additionally, it plots feature similarities for two example images.
