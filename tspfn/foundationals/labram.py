import tspfn.foundationals
import torch
from einops import rearrange
from modeling_vqnsp import vqnsp_encoder_base_decoder_3x200x12

def std_norm(self, x):
    mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
    std = torch.std(x, dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / std
    return x

if __name__ == "__main__":
    model = vqnsp_encoder_base_decoder_3x200x12(
        pretrained=True,
        pretrained_weights="/home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth",
        as_tokenizer=True,
        EEG_size=1000,
        n_code=8192,
        code_dim=32
    )
    x = torch.randn(4,17,1000)
    # Pad cls token
    # tokens = model.get_tokens(x, input_chans=16)
    x = rearrange(x, 'B N (A T) -> B N A T', T=200)
    input_chans = list(range(x.size(1)))
    quantize, embed_ind, emb_loss = model.encode(x, input_chans=input_chans)
    # print(tokens["token"].shape)
    print(f"token image is {embed_ind.view(x.size(0), -1).shape}")