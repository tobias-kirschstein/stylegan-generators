from jaxtyping import Float
from torch import nn, Tensor

from stylegan3_generator.generator import Generator


class StyleGAN3Generator(nn.Module):

    def __init__(self,
                 z_dim,
                 w_dim,
                 img_resolution,
                 img_channels,
                 mapping_kwargs={},
                 **synthesis_kwargs,
                 ):
        """
        Note, that by default the generated images are in the range [-64, 64] (default output_scale=0.25)
        To change that, use output_scale=...

        :param z_dim:
            Input latent (Z) dimensionality.
        :param w_dim:
            Intermediate latent (W) dimensionality.
        :param img_resolution:
            Output resolution.
        :param img_channels:
            Number of output color channels.
        :param mapping_kwargs:
            Arguments for MappingNetwork.
        :param synthesis_kwargs:
            Arguments for SynthesisNetwork.
        """
        super(StyleGAN3Generator, self).__init__()
        self._generator = Generator(z_dim,
                                    0,
                                    w_dim,
                                    img_resolution,
                                    img_channels,
                                    mapping_kwargs=mapping_kwargs,
                                    **synthesis_kwargs)

    def forward(self, code: Float[Tensor, "B Z"]) -> Float[Tensor, "B C R R"]:
        return self._generator(code, None)
