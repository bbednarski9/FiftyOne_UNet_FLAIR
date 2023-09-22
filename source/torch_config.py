'''
test_torch.py

Test PyTorch installation. In the terminal, run:
    python test_torch.py

Conda environment:
    med_web_llm
'''

import os
import torch

# setup pytorch and make sure that the correct GPU is selected
def torch_and_cuda_setup(seed):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU ID: {torch.cuda.get_device_name(0)}")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    return

def test_pytorch():
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.rand(2, 3))

if __name__ == "__main__":
    test_pytorch()
    print('Hello World!')