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
        EEG_size=1600,
        n_code=8192,
        code_dim=32
    )
    x = torch.randn(4,16,1600)
    # # tokens = model.get_tokens(x, input_chans=16)
    # x = rearrange(x, 'B N (A T) -> B N A T', T=200)
    # input_chans = list(range(16))
    # quantize, embed_ind, emb_loss = model.encode(x, input_chans=input_chans)
    # # print(tokens["token"].shape)
    # print(f"token image is {embed_ind.view(x.size(0), -1).shape}")
    with torch.no_grad():
        # If your model object is already loaded with 128-ch weights:
        # We pad to 128 just for the forward pass, then discard the rest.
        x_padded = F.pad(x, (0, 0, 0, 112)) 
        quantize, embed_ind, _ = model.encode(x_padded, input_chans=None)

    # 3. Get exactly the 128 tokens you want (16 ch * 8 patches)
    # embed_ind is shape [B, 1024]
    indices_3d = embed_ind.view(x.size(0), 128, 8)
    real_indices = indices_3d[:, :16, :] # Keep only the 16 channels you care about
    final_tokens = real_indices.flatten(1) # Shape: [4, 128]