# CAM Implementation

Class Activation Map(CAM) implementation by PyTorch. Paper can be seen in [here.](https://arxiv.org/abs/1512.04150)

Implementation detail is different from the original implementation from [this repository](https://github.com/zhoubolei/CAM)

You can execute this program by type this in shell

```
cd cam && bash do.sh
```


# Dependency & TroubleShooting


for installing overall depency of this repo, follow this installation : 

```
sudo apt update
sudo apt install build-essential
pip install chainercv
pip install matplotlib
pip install imageio

sudo apt-get remove cython
pip install -U cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

for spend some lazy times... use tmux.

```
sudo apt-get install tmux
```


# Acknowledgements

Last version of this repository has so many bugs, so it leads to borrow base codes from [IRN implementation](https://github.com/jiwoon-ahn/irn). Thanks for original repository writer. Applying denseCRF and AffinityNet(part of IRN) is heavily done from original repository.

## deprecated installation

use this command for treating the error on 'typing-extensions' : 

`pip install typing-extensions==4.3.0`

use this command for treating the error on 'importing ToTensor in albumentation'

`pip install albumentations==0.4.6`