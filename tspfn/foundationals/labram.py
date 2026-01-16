import sys
from pathlib import Path

# Add LaBraM to path before any labram-specific imports
labram_path = Path("submodules/labram").resolve()
sys.path.append(str(labram_path))
print(f"Added LaBraM path to sys.path: {labram_path}")
from modeling_vqnsp import vqnsp_encoder_base_decoder_3x200x12

if __name__ == "__main__":
    model = vqnsp_encoder_base_decoder_3x200x12(
        pretrained=True,
        pretrained_weights="/home/stympopper/pretrainingTSPFN/ckpts/labram_vqnsp.pth",
        as_tokenizer=True,
        EEG_size=1600,
        n_code=8192,
        code_dim=32
    )
    print(model)