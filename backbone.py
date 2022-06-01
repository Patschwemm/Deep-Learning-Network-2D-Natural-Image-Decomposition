import segmentation_models_pytorch as smp

# we take resnet34 as first approach with pretrained weights
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")