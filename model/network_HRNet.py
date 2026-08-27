from pathlib import Path

import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file

from attention.BAM import BAM
from attention.coordatt import CoordAtt
from model.DWConv import DWConv


class HRNetW18SmallV2Encoder(nn.Module):
    """ImageNet-pretrained HRNet with outputs adapted to the old ResNet contract."""

    pretrained_path = (
        Path(__file__).resolve().parent
        / "pretrained"
        / "hrnet_w18_small_v2.gluon_in1k.safetensors"
    )
    backbone_channels = (64, 128, 256, 512, 1024)
    output_channels = (64, 64, 128, 256, 512)
    output_reductions = (2, 4, 8, 16, 32)

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "hrnet_w18_small_v2",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        channels = tuple(self.backbone.feature_info.channels())
        reductions = tuple(self.backbone.feature_info.reduction())
        if channels != self.backbone_channels or reductions != self.output_reductions:
            raise RuntimeError(
                "Unexpected HRNet feature contract: "
                f"channels={channels}, reductions={reductions}"
            )
        if pretrained:
            self._load_local_pretrained()

        # Match the five feature maps expected by the existing Decoder:
        # [64, 128, 256, 512, 1024] -> [64, 64, 128, 256, 512].
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for in_channels, out_channels in zip(
                self.backbone_channels, self.output_channels
            )
        ])
        self._init_adapters()

    def _load_local_pretrained(self):
        if not self.pretrained_path.is_file():
            raise FileNotFoundError(
                f"HRNet pretrained weights not found: {self.pretrained_path}"
            )

        checkpoint = load_file(str(self.pretrained_path), device="cpu")
        model_state = self.backbone.state_dict()
        compatible_state = {
            name: value
            for name, value in checkpoint.items()
            if name in model_state and value.shape == model_state[name].shape
        }
        incompatible = self.backbone.load_state_dict(compatible_state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "Local HRNet checkpoint is incomplete; missing feature weights: "
                + ", ".join(incompatible.missing_keys[:10])
            )

    def _init_adapters(self):
        for adapter in self.adapters:
            nn.init.kaiming_normal_(adapter[0].weight, mode="fan_out", nonlinearity="relu")
            nn.init.ones_(adapter[1].weight)
            nn.init.zeros_(adapter[1].bias)

    def forward(self, x):
        features = self.backbone(x)
        return [adapter(feature) for adapter, feature in zip(self.adapters, features)]


class zh_net(nn.Module):
    def __init__(self, freeze_bn=False, pretrained=True):
        super(zh_net, self).__init__()
        # A/B share one HRNet encoder and therefore use exactly the same weights.
        self.encoder = HRNetW18SmallV2Encoder(pretrained=pretrained)
        self.decoder = Decoder()

        if freeze_bn:
            self.freeze_bn()

    def forward(self, A, B):
        output1 = self.encoder(A)
        output2 = self.encoder(B)
        result = self.decoder(output1, output2)
        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.de_block2 = DWConv(out_channels, out_channels)

        self.att = CoordAtt(out_channels, out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)

        self.de_block5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input1, input, input2):
        x0 = torch.cat((input1, input, input2), dim=1)
        x0 = self.de_block1(x0)
        x = self.de_block2(x0)
        x = self.att(x)
        x = self.de_block3(x)
        x = x + x0
        al = self.de_block4(x)
        result = self.de_block5(x)

        return al, result


class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        self.dir_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]).float())

    def forward(self, x, masks_pred, edge_pred):
        direc_pred = self.dir_head(x)
        direc_pred = direc_pred.softmax(1)
        edge_mask = 1 * (torch.sigmoid(edge_pred).detach() > 0.5)
        refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (
                    1 - edge_mask)
        return refined_mask_pred


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.bam = BAM(1024)
        self.db1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            DWConv(512, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.db2 = decoder_block(1024, 256)
        self.db3 = decoder_block(512, 128)
        self.db4 = decoder_block(256, 64)
        self.db5 = decoder_block(192, 32)

        self.classifier1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

        self.classifier2 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.interpo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = ref_seg()
        self._init_weight()

    def forward(self, input1, input2):
        input1_1, input2_1, input3_1, input4_1, input5_1 = input1[0], input1[1], input1[2], input1[3], input1[4]
        input1_2, input2_2, input3_2, input4_2, input5_2 = input2[0], input2[1], input2[2], input2[3], input2[4]

        x = torch.cat((input5_1, input5_2), dim=1)
        x = self.bam(x)
        x = self.db1(x)

        # 512*16*16
        al1, x = self.db2(input4_1, x, input4_2)  # 256*32*32
        al2, x = self.db3(input3_1, x, input3_2)  # 128*64*64
        al3, x = self.db4(input2_1, x, input2_2)  # 64*128*128
        al4, x = self.db5(input1_1, x, input1_2)  # 32*256*256

        edge = self.classifier1(x)
        seg = self.classifier2(torch.cat((x, self.interpo(al4)), 1))
        result = self.refine(x, seg, edge)

        return al1, al2, al3, al4, result, seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    test_data1 = torch.rand(2, 3, 256, 256).cuda()
    test_data2 = torch.rand(2, 3, 256, 256).cuda()
    test_label = torch.randint(0, 2, (2, 1, 256, 256)).cuda()

    model = zh_net()
    model = model.cuda()
    output = model(test_data1, test_data2)


