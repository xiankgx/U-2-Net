import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:],
                        mode='bilinear',
                        align_corners=True)
    return src


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=True):
        super(ConvBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm

        self.block = nn.Sequential(
            *([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), ]
              + ([nn.BatchNorm2d(out_channels), ] if batch_norm else [])
              + [nn.ReLU(), ])
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Unet(nn.Module):

    def __init__(self,
                 backbone,
                 in_channels,
                 out_channels=1,
                 decoder_channels=[512, 256, 128, 64],
                 decoder_block_repeats=[2, 2, 2, 2],
                 decoder_batch_norm=True):
        super(Unet, self).__init__()

        self.backbone = backbone

        assert len(decoder_channels) == len(in_channels) - 1
        assert len(decoder_channels) == len(decoder_block_repeats)

        upsample_projection_blocks = []
        decoder_blocks = []
        prev_channel = in_channels[-1]
        for i, c in enumerate(decoder_channels):
            upsample_projection_blocks.append(
                nn.Conv2d(prev_channel, c, 3, 1, 1))
            decoder_blocks.append(
                nn.Sequential(*(
                    [ConvBNReLU(c + in_channels[-2 - i], c,
                                batch_norm=decoder_batch_norm)]
                    + [ConvBNReLU(c, c, batch_norm=decoder_batch_norm)
                       for _ in range(decoder_block_repeats[i])]
                ))
            )

            prev_channel = c

        self.upsample_projection_blocks = nn.ModuleList(
            upsample_projection_blocks)
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.final_conv = nn.Conv2d(decoder_channels[-1], out_channels,
                                    3, 1, 1)

    def forward(self, x, y=None):
        if y is not None:
            backbone_outputs = self.backbone(x, y)
        else:
            backbone_outputs = self.backbone(x)

        x = backbone_outputs[-1]
        side_outputs = [x, ]
        for i, m in enumerate(self.decoder_blocks):
            x_up = _upsample_like(x, backbone_outputs[-2 - i])
            x_up = self.upsample_projection_blocks[i](x_up)

            x = torch.cat([x_up, backbone_outputs[-2 - i]], dim=1)

            x = m(x)
            side_outputs.append(x)

        x = self.final_conv(x)
        return tuple([x, ] + side_outputs)


class UnetBackbone(nn.Module):

    def __init__(self,
                 input_channels=3,
                 channels=[64, 128, 256, 512, 1024],
                 encoder_block_repeats=[2, 2, 2, 2, 2],
                 batch_norm=True):
        super(UnetBackbone, self).__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.encoder_block_repeats = encoder_block_repeats
        self.batch_norm = batch_norm

        encoder_blocks = []
        prev_channel = input_channels
        for i, c in enumerate(channels):
            encoder_blocks.append(
                nn.Sequential(*(
                    [ConvBNReLU(prev_channel, c,
                                batch_norm=batch_norm)]
                    + [ConvBNReLU(c, c,
                                  batch_norm=batch_norm)
                       for _ in range(encoder_block_repeats[i])]
                ))
            )
            prev_channel = c
        self.layers = nn.ModuleList(encoder_blocks)

    def forward(self, x):
        outputs = []
        for i, m in enumerate(self.layers):
            x = m(x)
            outputs.append(x)

            if i != len(self.layers) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        return outputs


class UnetBackboneMultiInputs(nn.Module):

    def __init__(self,
                 input_channels=3,
                 channels=[64, 128, 256, 512, 1024],
                 side_channels=[3, 64, 128, 256, 512],
                 encoder_block_repeats=[2, 2, 2, 2, 2],
                 batch_norm=True):
        super(UnetBackboneMultiInputs, self).__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.side_channels = side_channels
        self.encoder_block_repeats = encoder_block_repeats
        self.batch_norm = batch_norm

        encoder_blocks = []
        prev_channel = input_channels
        for i, c in enumerate(channels):
            encoder_blocks.append(
                nn.Sequential(*(
                    [ConvBNReLU(prev_channel + side_channels[i], c,
                                batch_norm=batch_norm)]
                    + [ConvBNReLU(c, c,
                                  batch_norm=batch_norm)
                       for _ in range(encoder_block_repeats[i])]
                ))
            )
            prev_channel = c
        self.layers = nn.ModuleList(encoder_blocks)

    def forward(self, x, y):
        outputs = []
        for i, m in enumerate(self.layers):
            # print(f">>> x.shape: {x.shape}")
            # print(f">>> y[i].shape: {y[i].shape}")
            x = m(torch.cat([x, y[i]], dim=1))
            outputs.append(x)

            if i != len(self.layers) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)

        return outputs


class ResNet50Backbone(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()

        self.extract_layers = [2, 4, 5, 6, 7]

        backbone = models.resnet50(pretrained=pretrained)
        self.layers = nn.ModuleList(list(backbone.children())[:-2])

    def forward(self, x):
        outputs = [x, ]

        for i, m in enumerate(self.layers):
            x = m(x)
            # print(f"RN50 layer #{i} shape: {x.shape}")

            if i in self.extract_layers:
                outputs.append(x)

        return outputs


class CustomNet(nn.Module):

    def __init__(self, out_channels=1):
        super(CustomNet, self).__init__()

        self.deep_features = ResNet50Backbone()
        deep_features_channels = 2048

        # Standard Unet with side additional side inputs at every encoder blocks
        segnet_backbone = UnetBackboneMultiInputs(
            channels=[32, 64, 128, 256, 512],
            encoder_block_repeats=[3, 3, 3, 3, 3],
            side_channels=[3, 32, 64, 128, 256]
        )

        self.deep_projections = nn.ModuleList([
            nn.Conv2d(deep_features_channels,
                      segnet_backbone.side_channels[0], 1),
            nn.Conv2d(deep_features_channels,
                      segnet_backbone.side_channels[1], 1),
            nn.Conv2d(deep_features_channels,
                      segnet_backbone.side_channels[2], 1),
            nn.Conv2d(deep_features_channels,
                      segnet_backbone.side_channels[3], 1),
            nn.Conv2d(deep_features_channels,
                      segnet_backbone.side_channels[4], 1)
        ])

        self.segnet = Unet(backbone=segnet_backbone,
                           in_channels=segnet_backbone.channels,
                           decoder_channels=[256, 128, 64, 32],
                           decoder_block_repeats=[3, 3, 3, 3])
        # For upsampling predictions to full size
        self.side_projections = nn.ModuleList([
            nn.Conv2d(segnet_backbone.channels[4], out_channels,
                      3, 1, 1),
            nn.Conv2d(segnet_backbone.channels[3], out_channels,
                      3, 1, 1),
            nn.Conv2d(segnet_backbone.channels[2], out_channels,
                      3, 1, 1),
            nn.Conv2d(segnet_backbone.channels[1], out_channels,
                      3, 1, 1),
            nn.Conv2d(segnet_backbone.channels[0], out_channels,
                      3, 1, 1)
        ])
        self.fuse = nn.Conv2d(len(self.side_projections)
                              * out_channels, out_channels, 1)

    def forward(self, x):
        input = x

        # Extract deep features from a deep model
        deep_features = self.deep_features(input)

        # Project last deep feature map to various sizes suitable as side input to Unet
        projected_deep_features = []
        for i, m in enumerate(self.deep_projections):
            target_size = (
                int(x.size(2)/(2 ** i)),
                int(x.size(3)/(2 ** i))
            )
            # print(f"target_size: {target_size}")
            _x = F.interpolate(deep_features[-1],
                               size=target_size,
                               mode="bilinear",
                               align_corners=True)
            _x = m(_x)
            projected_deep_features.append(_x)

        # Make prediction with Unet
        seg_outs = self.segnet(input, projected_deep_features)

        # Resize and project side outputs
        side_outs = []
        for i, t in enumerate(seg_outs[1:]):
            t = _upsample_like(t, seg_outs[-1])
            t = torch.sigmoid(self.side_projections[i](t))
            side_outs.append(t)

        # Fuse all side outputs
        fused = torch.sigmoid(self.fuse(torch.cat(side_outs, dim=1)))

        outputs = [fused, ] + side_outs
        return tuple(outputs)


if __name__ == "__main__":
    input = torch.rand(2, 3, 1020, 235)

    rn50 = ResNet50Backbone()
    outputs = rn50(input)
    for i, t in enumerate(outputs):
        print(f"rn50 output #{i}", t.shape)

    model = CustomNet()
    outputs = model(input)
    for i, t in enumerate(outputs):
        print(f"model output #{i}", t.shape)

    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("model parameters:", num_params)

    torch.save(model.state_dict(), "custom.pth")
