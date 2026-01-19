import tspfn.foundationals
import torch
import numpy
import argparse
from einops import rearrange
from modeling_vqnsp import vqnsp_encoder_base_decoder_3x200x12
from modeling_pretrain import labram_base_patch200_1600_8k_vocab

torch.serialization.add_safe_globals([
    numpy.dtypes.Float64DType, 
    numpy.core.multiarray.scalar,
    numpy.dtype,
    argparse.Namespace
])

class TimeSeriesNeuralTokenizer(torch.nn.Module):
    def __init__(self, pretrained_weight: str = None, ts_size: int = 1000):
        super().__init__()
        self.model = vqnsp_encoder_base_decoder_3x200x12(
            pretrained=True,
            pretrained_weight=pretrained_weight,
            as_tokenzer=True,
            EEG_size=ts_size,
            n_code=8192,
            code_dim=64,
        )

    def forward(self, x: torch.Tensor, input_chans: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, T) Time series input.
            input_chans: List of input channels to consider for tokenization.
        Returns:
            embed_ind: (B, N, num_tokens) Token indices.
        """
        B, N, T = x.size()
        assert T % 200 == 0, "Time dimension must be divisible by 200."
        A = T // 200
        x = rearrange(x, "B N (A T) -> B N A T", A=A)
        input_chans = list(range(x.size(1)+1))  # +1 for cls token
        quantize, embed_ind, emb_loss = self.model.encode(x, input_chans=input_chans)
        # Decoder
        decoded_output = self.model.decoder(quantize, input_chans=input_chans)
        return decoded_output

class TimeSeriesPatchEmbedder(torch.nn.Module):
    def __init__(self, pretrained_weights: str = None):
        super().__init__()
        transformerMEM = labram_base_patch200_1600_8k_vocab(
            pretrained= pretrained_weights is not None,
            init_ckpt=pretrained_weights,
        )

        student = transformerMEM.student

        self.patch_embed = student.patch_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T) Time series input.
        Returns:
            embed_ind: (B, N, num_tokens) Token indices.
        """
        B, N, T = x.size()
        assert T % 200 == 0, "Time dimension must be divisible by 200."
        A = T // 200
        x = rearrange(x, "B N (A T) -> B N A T", A=A)
        tokens = self.patch_embed(x)
        return tokens

if __name__ == "__main__":
    transformerMEM = labram_base_patch200_1600_8k_vocab(
        pretrained=False,
        # init_ckpt="/home/stympopper/pretrainingTSPFN/ckpts/labram-base.pth",
        init_values=0.1,
    )
    ckpt_path = "/home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth"
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    msg = transformerMEM.load_state_dict(checkpoint["model"], strict=False)
    print(f"Load ckpt msg: {msg}")
    student = transformerMEM.student
    model = student.patch_embed
    x = torch.randn(4, 16, 1000)
    x = rearrange(x, "B N (A T) -> B N A T", T=200)
    tokens = model(x)
    print(f"token shape is {tokens.shape}")
#     model = vqnsp_encoder_base_decoder_3x200x12(
#         pretrained=True,
#         pretrained_weight="/home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth",
#         as_tokenzer=True,
#         EEG_size=1000,
#         n_code=8192,
#         code_dim=64,
#     )
#     x = torch.randn(4, 16, 1000)
#     x = rearrange(x, "B N (A T) -> B N A T", T=200)
#     input_chans = list(range(x.size(1)+1))
#     quantize, embed_ind, emb_loss = model.encode(x, input_chans=input_chans)
#     # decoded_output = model.decoder(quantize, input_chans=input_chans)
#     # print(f"quantize shape is {quantize.shape}")
#     # print(f"embed_ind shape is {embed_ind.shape}")
#     # print(f"decoded output shape is {decoded_output.shape}")
#     # # Random select channels
#     # indices = torch.randperm(quantize.size(-2))[:4]
#     # quantize = quantize[:, :, indices, :]
#     # Pool quantize on channels
#     quantize = quantize.mean(dim=2, keepdim=True)
#     print(f"quantize before rearrange is {quantize.shape}")
#     quantize = rearrange(quantize, "B D C A -> B (A C) D")
#     # print(tokens["token"].shape)
#     print(f"token image is {embed_ind.view(x.size(0), -1).shape}")
#     print(f"quantize is {quantize.flatten(start_dim=1).shape}")
