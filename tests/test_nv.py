import torch
import segmentation_models_pytorch as smp

if __name__=='__main__':

    model_resnet18 = smp.Unet(
        encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=3,  # model output channels (number of classes in your dataset)
    )

    pred = model_resnet18(torch.randn((1, 1, 576, 576)))
    print(pred)