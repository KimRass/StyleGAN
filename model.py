# Reference: https://github.com/SiskonEmilia/StyleGAN-PyTorch/blob/master/model.py
# "We feed a dedicated noise image to each layer of the synthesis network. The noise image is broadcasted to all feature maps using learned perfeature scaling factors and then added to the output of the corresponding convolution."

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0; 1). 


def generate_latent_vec(
    batch_size: int, latent_dim: int = 512,
) -> TensorType["batch_size", "latent_dim"]:
    """Generate a random latent vector $\mathbf{z} \in \mathcal{Z}$.
    """
    return torch.randn(batch_size, latent_dim)


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim: int = 512, num_layers: int = 8):
        """Mapping network $f$.
        "Given a latent code z in the input latent space Z, a non-linear mapping network f : Z ! W first produces w 2 W."
        "We set the dimensionality of both spaces to 512, and the mapping f is implemented using an 8-layer MLP.
        "The mapping network $f$ consists of 8 layers."
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Fully connected layers with leaky ReLU activations
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*layers)

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     """
    #     Initialize the weights of the fully connected layers.
    #     StyleGAN uses a scaled He initialization.
    #     """
    #     for m in self.mapping:
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
    #             nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize the input latent vector
        x = F.normalize(x, dim=1)  # L2 normalization of z
        return self.layers(x)


class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class LearnedAffineTransformation(nn.Module):
    """
    "Learned affine transformations then specialize w to styles y = (ys, yb) that control adaptive instance normalization (AdaIN) operations after each convolution layer of the synthesis network $g$."
    """
    def __init__(self, dim_latent, channels):
        super().__init__()

        self.transform = SLinear(dim_latent, channels * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[: channels] = 1
        self.transform.linear.bias.data[channels:] = 0
        # "The biases and noise scaling factors are initialized to zero, except for the
        # biases associated with ys that we initialize to one."

    def forward(self, w):
        # Gain scale factor and bias with:
        return self.transform(w).unsqueeze(2).unsqueeze(3)


class AdaIn(nn.Module):
    """
    adaptive instance normalization
    """
    def __init__(self, channels):
        super().__init__()

        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, x, style):  # `style`: $y$.
        scale, bias = torch.chunk(style, chunks=2, dim=1)
        return scale * self.norm(x) + bias


class LearnedPerChannelScalingFactor(nn.Module):
    '''
    Learned per-channel scale factor, used to scale the noise
    '''
    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros((1, channels, 1, 1)))  # zeros??
    
    def forward(self, x):
        return self.weight * x


class Block1(nn.Module):
    '''
    This is the very first block of generator that get the constant value as input
    '''
    def __init__ (self, latent_dim: int, const_size: int = 4, channels: int = 512):
        super().__init__()

        self.const = nn.Parameter(torch.zeros(1, channels, const_size, const_size))
        # "The constant input in synthesis network is initialized to one."
        self.scaling_factor1 = quick_scale(LearnedPerChannelScalingFactor(channels))
        self.affine_transform1 = LearnedAffineTransformation(latent_dim, channels)
        self.adain  = AdaIn(channels)
        self.lrelu  = nn.LeakyReLU(0.2)
        self.conv = SConv2d(channels, channels, 3, padding=1)
        self.scaling_factor2 = quick_scale(LearnedPerChannelScalingFactor(channels))
        self.affine_transform2 = LearnedAffineTransformation(latent_dim, channels)
        # Convolutional layer
    
    def forward(self, w, noise):
        # Gaussian Noise: Proxyed by generator
        # scaling_factor1 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()
        # scaling_factor2 = torch.normal(mean=0,std=torch.ones(self.constant.shape)).cuda()
        x = self.const.repeat(noise.size(0), 1, 1, 1)
        x = x + self.scaling_factor1(noise)
        x = self.adain(x, style=self.affine_transform1(w))
        x = self.lrelu(x)
        x = self.conv(x)
        x = x + self.scaling_factor2(noise)
        # "Gaussian noise is added after each convolution, before evaluating the nonlinearity."
        x = self.adain(x, style=self.affine_transform2(w))
        x = self.lrelu(x)
        return x


class Block2(nn.Module):
    '''
    This is the general class of style-based convolutional blocks
    '''
    def __init__ (self, latent_dim: int, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = SConv2d(in_channels, out_channels, 3, padding=1)
        self.scaling_factor1 = quick_scale(LearnedPerChannelScalingFactor(out_channels))
        self.affine_transform1 = LearnedAffineTransformation(
            latent_dim, out_channels,
        )
        self.adain = AdaIn(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = SConv2d(out_channels, out_channels, 3, padding=1)
        self.scaling_factor2 = quick_scale(LearnedPerChannelScalingFactor(out_channels))
        self.affine_transform2 = LearnedAffineTransformation(
            latent_dim, out_channels,
        )
    
    def forward(self, x, w, noise):
        # Upsample: Proxyed by generator
        # result = nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
        #                                           align_corners=False)
        # Conv 3*3
        x = self.conv1(x)
        # Gaussian Noise: Proxyed by generator
        # scaling_factor1 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # scaling_factor2 = torch.normal(mean=0,std=torch.ones(result.shape)).cuda()
        # Conv & Norm
        x = x + self.scaling_factor1(noise)
        x = self.adain(x, style=self.affine_transform1(w))
        x = self.lrelu(x)
        x = self.conv2(x)
        x = x + self.scaling_factor2(noise)
        x = self.adain(x, style=self.affine_transform2(w))
        x = self.lrelu(x)
        return x


class ToRGB(nn.Module):
    """
    "The output of the last layer is converted to RGB using a separate 1 × 1 convolution"
    """
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class StyleGANSynthesisNetwork(nn.Module):
    """Synthesis network $g$.
    "The synthesis network $g$ consists of 18 layers—two for each resolution (42 􀀀 10242)."
    "Our generator has a total of 26.2M trainable parameters"
    """
    def __init__ (self,
            latent_dim: int,
            num_mapping_net_layers: int = 8,
            const_size: int = 4,
            channels: int = 512,
        ):
        super().__init__()

        self.mapping_net = MappingNetwork(
            latent_dim=latent_dim, num_layers=num_mapping_net_layers,
        )
        self.res4_block = Block1(
            latent_dim=latent_dim, const_size=const_size, channels=channels,
        )
        self.res8_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res16_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res32_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res64_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res128_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res256_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res512_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.res1024_block = Block2(
            latent_dim=latent_dim, in_channels=channels, const_size=channels,
        )
        self.to_rgb = ToRGB(in_channels=channels)

    def forward(self, latent_vec, noise):
        w = self.mapping_net(latent_vec)  # $\mathbf{w}$.
        x = self.res4_block(w=w, noise=noise)
        x = self.res8_block(x, w=w, noise=noise)
        x = self.res16_block(x, w=w, noise=noise)
        x = self.res32_block(x, w=w, noise=noise)
        x = self.res64_block(x, w=w, noise=noise)
        x = self.res128_block(x, w=w, noise=noise)
        x = self.res256_block(x, w=w, noise=noise)
        x = self.res512_block(x, w=w, noise=noise)
        x = self.res1024_block(x, w=w, noise=noise)
        return self.to_rgb(x)


if __name__ == "__main__":
    latent_dim = 512
    num_layers = 8
    batch_size = 16

    # Instantiate the mapping network
    mapping_net = MappingNetwork(latent_dim=latent_dim, num_layers=num_layers)

    z = generate_latent_vec(batch_size=batch_size, latent_dim=latent_dim)

    # Get the transformed latent vector w
    w = mapping_net(z)
    print("Input z shape:", z.shape)
    print("Output w shape:", w.shape)
