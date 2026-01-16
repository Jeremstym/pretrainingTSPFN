import tspfn.foundationals
import torch
from modeling_vqnsp import vqnsp_encoder_base_decoder_3x200x12

if __name__ == "__main__":
    model = vqnsp_encoder_base_decoder_3x200x12(
        pretrained=True,
        pretrained_weights="/home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth",
        as_tokenizer=True,
        EEG_size=1000,
        n_code=8192,
        code_dim=32
    )
    x = torch.randn(4,16,8,200)
    tokens = model.get_tokens(x, input_chans=16)
    print(tokens["token"].shape)