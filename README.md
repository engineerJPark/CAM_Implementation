# CAM Implementation

Class Activation Map(CAM) implementation by PyTorch. Paper can be seen in [here.](https://arxiv.org/abs/1512.04150)

Implementation detail is different from the original implementation from [this repository](https://github.com/zhoubolei/CAM)


# About TroubleShooting

use this command for treating the error on 'typing-extensions' : 

`pip install typing-extensions==4.3.0`

use this command for treating the error on 'importing ToTensor in albumentation'

`pip install albumentations==0.4.6`

for installing chainercv, follow this installation : 

`
sudo apt update
sudo apt install build-essential
pip install chainercv
`

for spend some lazy times... use tmux.

`sudo apt-get install tmux`


# Acknowledgements

Last version of this repository has so many bugs, so it leads to borrow base codes from [IRN implementation](https://github.com/jiwoon-ahn/irn). Applying denseCRF and AffinityNet is heavily done from original repository.