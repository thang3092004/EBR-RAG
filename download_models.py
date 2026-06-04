import os
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

# Setup robust flash_attn mock to prevent importlib.util.find_spec errors
flash_attn_spec = ModuleSpec("flash_attn", None)
flash_attn_mock = MagicMock()
flash_attn_mock.__spec__ = flash_attn_spec
flash_attn_mock.__path__ = []
sys.modules["flash_attn"] = flash_attn_mock
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()
sys.modules["flash_attn.bert_padding"] = MagicMock()


def download_models():
    print("=== Downloading MiniCPM-V-2_6-int4 ===")
    try:
        from transformers import AutoModel, AutoTokenizer
        # This will download and cache the model from HuggingFace
        model_name = "openbmb/MiniCPM-V-2_6-int4"
        AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation="sdpa")
        print("âœ… MiniCPM-V-2_6-int4 downloaded successfully.")
    except Exception as e:
        print(f"âŒ Failed to download MiniCPM: {e}")
        print("Please ensure you have `transformers` installed and internet connection.")

    print("\n=== Downloading ImageBind ===")
    try:
        # Check if checkpoints/imagebind_huge.pth is corrupted/incomplete (less than 4GB)
        pth_path = os.path.join(".checkpoints", "imagebind_huge.pth")
        if os.path.exists(pth_path):
            file_size = os.path.getsize(pth_path)
            if file_size < 4000000000:  # ~4GB threshold
                print(f"âš ï¸ Truncated weight file detected ({file_size / (1024*1024):.1f} MB). Deleting to restart download...")
                try:
                    os.remove(pth_path)
                except Exception as del_err:
                    print(f"Failed to remove corrupted file: {del_err}")

        from imagebind.models import imagebind_model
        # This will download imagebind_huge.pth to .checkpoints/
        imagebind_model.imagebind_huge(pretrained=True)
        print("âœ… ImageBind downloaded successfully.")
    except Exception as e:
        print(f"âŒ Failed to download ImageBind: {e}")
        print("Please ensure `imagebind` is installed in your environment.")

if __name__ == "__main__":
    download_models()
