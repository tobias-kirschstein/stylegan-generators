from jaxtyping import Float
from torch import nn, Tensor

from stylegan_generators.stylegan3 import Generator as GeneratorS3
from stylegan_generators.stylegan2 import Generator as GeneratorS2


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
        self._generator = GeneratorS3(z_dim,
                                      0,
                                      w_dim,
                                      img_resolution,
                                      img_channels,
                                      mapping_kwargs=mapping_kwargs,
                                      **synthesis_kwargs)

        self._params_std = 0.2
        self.reset_parameters()

    def _init_mapping_weights(self, params: nn.Parameter):
        nn.init.normal_(params, 0.0,
                        self._params_std * 100)  # Mapping has lr_multiplier 0.01, hence the weights have to be 100x

    def _init_input_affine_weights(self, params: nn.Parameter):
        nn.init.zeros_(params)

    def _init_input_conv_weights(self, params: nn.Parameter):
        nn.init.normal_(params, 0.0, self._params_std)

    def _init_input_affine_bias_weights(self, params: nn.Parameter):
        # Bias init: bias_init=[1,0,0,0]?
        pass

    def _init_affine_weights(self, params: nn.Parameter):
        nn.init.normal_(params, 0.0, self._params_std)

    def _init_conv_weights(self, params: nn.Parameter):
        nn.init.normal_(params, 0.0, self._params_std)

    def reset_parameters(self):
        for key, param in self.named_parameters():
            # Mapping network
            if 'mapping' in key and 'weight' in key:
                self._init_mapping_weights(param)

            # Input Network
            elif 'input.affine.weight' in key:
                self._init_input_affine_weights(param)
            elif 'input.weight' in key:
                self._init_input_conv_weights(param)
            elif 'input.affine.bias' in key:
                self._init_input_affine_bias_weights(param)

            # Conv layers
            elif 'affine.weight' in key:
                self._init_affine_weights(param)
            elif 'weight' in key:
                self._init_conv_weights(param)

    def forward(self, code: Float[Tensor, "B Z"]) -> Float[Tensor, "B C R R"]:
        return self._generator(code, None)


class StyleGAN2Generator(nn.Module):

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
        super(StyleGAN2Generator, self).__init__()
        self._generator = GeneratorS2(z_dim,
                                      0,
                                      w_dim,
                                      img_resolution,
                                      img_channels,
                                      mapping_kwargs=mapping_kwargs,
                                      **synthesis_kwargs)

    def forward(self, code: Float[Tensor, "B Z"]) -> Float[Tensor, "B C R R"]:
        return self._generator(code, None)
