import torch
import math
from torch import nn
from torch._C import device
try:
    from TimeSeriesProcessing.models.transformer.components.layers import Encoder, EncoderLayer, ConvLayer
    from TimeSeriesProcessing.models.transformer.components.selfattention import FullAttention, AttentionLayer
    from TimeSeriesProcessing.models.transformer.components.encoding import PositionalEncoding
except:
    from components.layers import Encoder, EncoderLayer, ConvLayer
    from components.selfattention import FullAttention, AttentionLayer
    from components.encoding import PositionalEncoding


class TransformerTimeSeries(nn.Module):
    """Transformer model adapted to time series data

    This model follows the implementation from [1], which adapted the original
    model from [2] to perform a time series prediction task instead of the original
    task of NLP. The adaptation to the time series task removed the input
    embedding from the encoder but maintained the positional encoding process.

    References:
    [1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural
        information processing systems. 2017.
    [2] Wu, Neo, et al. "Deep transformer models for time series forecasting: The influenza
        prevalence case." arXiv preprint arXiv:2001.08317 (2020).

    Args:
        device            : Torch device to process the model
        encoder_vector_sz : Size of the encoder input vector, which also define the
                            number the encoder's input features.
        decoder_vector_sz : Size of the decoder input vector, which also define the
                            number the decoder's input features.
        encoder_mask      : Either use or not a mask to avoid look ahead in the encoder.
        decoder_mask      : Either use or not a mask to avoid look ahead in the decoder.
        num_layers        : Number of decoder and encoder layers.
        nhead             : Number of parallel multi attention heads.
        n_time_steps      : Number of timesteps used as inputs in the encoder and
                            decoder.
        d_model           : Model's vector dimmension.
        max_len : Array of objects with simulation data

    """

    def __init__(self,
                 device,
                 encoder_vector_sz=1,
                 encoder_mask=False,
                 n_enc_layers=1,
                 nhead=8,
                 n_encoder_time_steps=30,
                 output_vector_sz=1,
                 d_model=100,
                 dropout=0.5,
                 iterative=True,
                 classification=False):

        super(TransformerTimeSeries, self).__init__()
        self.model_type = 'Transformer'
        self.device = device
        self.encoder_mask = encoder_mask
        self.n_encoder_time_steps = n_encoder_time_steps
        self.output_vector_sz = output_vector_sz
        self.iterative = iterative
        self.classification = classification
        # Positional encoding used in the decoder and encoder
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder and encoder layers
        # Note that this implementation follows the data input format of
        # (batch, timestamps, features).

        factor = 5
        activation = 'relu'
        output_attention = True
        d_ff = 2048
        self.transformer_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, factor=factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, nhead),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Encoder input projection
        self.encoder_projection = nn.Linear(encoder_vector_sz, d_model)

        # Transformer output layer
        # Note that the output format of this model is (batch, pred_steps, output_vector_sz), where
        # as a time series predictor the pred_steps contain the estimation of future
        # behavior of one variable in time.
        self.linear = nn.Linear(d_model, self.output_vector_sz, bias=True)
        self.linear_2 = nn.Linear(self.output_vector_sz, self.output_vector_sz, bias=True)
        self.out_fcn = nn.Tanh()
        self.drop = nn.Dropout(p=0.2)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz, in_sz, iterative=True):
        """Generates decoder mask

        This mask prevents the look ahead behavior in the decoder or encoder
        process.

        Args:
            sz : Size of timestep matrix mask.
        """
        # Provides mask for multi-step iterative scenarios
        if iterative:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0,
                                            float('-inf')).masked_fill(
                                                mask == 1, float(0.0))

        # Provides mask for the multi-step direct scenarios
        else:
            mask = torch.zeros((sz, sz))
            mask[:, in_sz:] = 1
            # mask[in_sz:] = 1

            mask = mask.float().masked_fill(mask == 1,
                                            float('-inf')).masked_fill(
                                                mask == 0, float(0.0))

        return mask

    def encoder_process(self, src, output_attention=False):
        """Encoder process of the Transformer network.

        According to [1], the tranformer encoder produces an encoded version
        of the entry series with the positional information and a self learned
        projection of the input data, generating a encoder matrix of dmodel
        dimension.

        Args:
            src: Input data of  the encoder, with shape
                 (batch, n_time_steps, encoder_vector_sz)
        """

        x = src
        x = self.encoder_projection(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        if self.encoder_mask:
            # Mask to avoid look ahead behavior in the encoder process
            mask = self._generate_square_subsequent_mask(
                self.n_encoder_time_steps,
                self.n_encoder_time_steps,
                iterative=self.iterative).to(self.device)
        else:
            mask = None

        x, attn = self.transformer_encoder(x, attn_mask=mask)
        # x = x.permute(1, 0, 2)
        if output_attention:
            return x, attn
        else:
            return x

    def init_weights(self):
        """Initialize the output layer weights.
        """
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        """Send the information through the transformer network.
        """

        src = x
        src = self.encoder_process(src)
        if self.classification:
            src = src[:, -1, :]
            out = self.linear(src)
            # out = self.out_fcn(out)
            # out = self.linear_2(out)
            # out = self.out_fcn(out)
        else:
            src = src[:, -1, :]
            out = self.linear(src)

        # out = torch.reshape(out, (-1, out.shape[1]))

        return out

    def encoder_attention(self, src, layer_idx=0):
        """Return the attention matrix given an x input

        Args:
            src : Input tensor with shape (batch, timesteps, entry_feat_dim)
        """
        _, attn = self.encoder_process(src, output_attention=True)

        return attn


def main():
    device = torch.device('cpu')
    print('Using device:', device)
    model = TransformerTimeSeries(device,
                                  n_encoder_time_steps=120,
                                  encoder_mask=False,
                                  output_vector_sz=2,
                                  encoder_vector_sz=1,
                                  iterative=False,
                                  classification=False).to(device)
    src = torch.rand(60, 120, 1).to(device)

    out = model(src)
    print(out.shape)


if __name__ == '__main__':
    main()
