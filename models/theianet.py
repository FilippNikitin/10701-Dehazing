import torch

from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Encoder, self).__init__()
        encoder = []

        for input_channel, output_channel in zip(input_channels, output_channels):
            convrelu = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
            encoder.append(convrelu)
        self.encoder = nn.ModuleList(encoder)

    def forward(self, x):
        output = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i != len(self.encoder) - 1:
                x = F.max_pool2d(x, kernel_size=2)
            output.append(x)
        return output


class BottleNeckEnhuncer(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(BottleNeckEnhuncer, self).__init__()
        self.convrelu_global = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.convrelu_avg_8 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.convrelu_avg_4 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.convrelu_max_2 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, w, h = x.shape

        glob = F.avg_pool2d(x, kernel_size=(w, h))
        x8 = F.avg_pool2d(x, kernel_size=(8, 8))
        x4 = F.avg_pool2d(x, kernel_size=(4, 4))
        x2 = F.max_pool2d(x, kernel_size=(2, 2))

        glob = self.convrelu_global(glob)
        x8 = self.convrelu_avg_8(x8)
        x4 = self.convrelu_avg_4(x4)
        x2 = self.convrelu_max_2(x2)

        glob = F.interpolate(glob, size=(w, h), mode='bilinear')
        x8 = F.interpolate(x8, size=(w, h), mode='bilinear')
        x4 = F.interpolate(x4, size=(w, h), mode='bilinear')
        x2 = F.interpolate(x2, size=(w, h), mode='bilinear')

        out = torch.cat([glob, x8, x4, x2], dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels[0], output_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(input_channels[1], output_channels[1], kernel_size=3, stride=1, padding=1)

    def forward(self, enhuncer_output, encoder_output):
        out1 = self.conv1(enhuncer_output)
        _, _, w, h = encoder_output.shape
        out = F.interpolate(out1, size=(w, h), mode='bilinear')

        x = torch.cat([out, encoder_output], dim=1)
        out2 = self.conv2(x)
        return out1, out2


class AggregationHead(nn.Module):
    def __init__(self, total_inp_channel, feat_channel, output_channel, scale_factors):
        super().__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(feat_channel, feat_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(total_inp_channel, feat_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ConvTranspose2d(feat_channel, feat_channel, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.scale_factors = scale_factors

    def forward(self, en1, en2, dec1, dec2):
        _, _, w, h = en1.shape
        w, h = w * self.scale_factors[0], h * self.scale_factors[0]

        en1 = F.interpolate(en1, size=(w, h), mode='bilinear')
        en2 = F.interpolate(en2, size=(w, h), mode='bilinear')
        dec1 = F.interpolate(dec1, size=(w, h), mode='bilinear')
        dec2 = F.interpolate(dec2, size=(w, h), mode='bilinear')
        x = torch.cat([en1, en2, dec1, dec2], dim=1)
        x = self.deconv(x)
        x = self.convrelu(x)
        return x


class TheiaNet(nn.Module):
    def __init__(self, in_chans=3, encoder_feats=16, enhuncer_feats=16, decoder_feats=16, head_feats=16):
        super().__init__()
        self.encoder = Encoder([in_chans, encoder_feats, encoder_feats],
                               [encoder_feats, encoder_feats, encoder_feats])
        self.enhuncer = BottleNeckEnhuncer(encoder_feats, enhuncer_feats)
        self.decoder = Decoder([enhuncer_feats * 4, encoder_feats + decoder_feats],
                               [decoder_feats, decoder_feats])
        self.aggregation_head = AggregationHead(2 * encoder_feats + 2 * decoder_feats, head_feats, in_chans,
                                                [2, 4, 2, 2])

    def forward(self, x):
        enc_out = self.encoder(x)
        enhuncer_out = self.enhuncer(enc_out[-1])
        dec_out = self.decoder(enhuncer_out, enc_out[0])
        out = self.aggregation_head(*enc_out[:2], *dec_out)
        return out
