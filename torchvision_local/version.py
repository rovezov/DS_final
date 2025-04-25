__version__ = '0.22.0a0+5f03dc5'
git_version = '5f03dc524bdb7529bb4f2e84d2d8c237233fc62a'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
