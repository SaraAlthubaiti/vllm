import torch
from vllm.model_executor.models.beats import BeatsEncoderWrapper

cfg = {
    "beats_ckpt": "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
}

model = BeatsEncoderWrapper(cfg)
dummy_audio = torch.randn(1, 16000)  # 1 second fake input
features = model(dummy_audio)

print("✅ Output shape:", features.shape)