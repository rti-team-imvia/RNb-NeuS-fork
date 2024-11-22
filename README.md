# RNb-NeuS
This is the official implementation of **RNb-NeuS: Reflectance and Normal-based Multi-View 3D Reconstruction**.

[Baptiste Brument*](https://bbrument.github.io/),
[Robin Bruneau*](https://robinbruneau.github.io/),
[Yvain Quéau](https://sites.google.com/view/yvainqueau),
[Jean Mélou](https://www.irit.fr/~Jean.Melou/),
[François Lauze](https://loutchoa.github.io/),
[Jean-Denis Durou](https://www.irit.fr/~Jean-Denis.Durou/),
[Lilian Calvet](https://scholar.google.com/citations?user=6JewdrMAAAAJ&hl=en)

### [Project page](https://robinbruneau.github.io/publications/rnb_neus.html) | [Paper](https://arxiv.org/abs/2312.01215)

<img src="assets/pipeline.png">

----------------------------------------
## Installation

dvd(2024/09/25): Python version: 3.9.13

```shell
git clone https://github.com/bbrument/RNb-NeuS.git
cd RNb-NeuS
pyenv local 3.11.3 # If using pyenv only
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python-headless
pip install trimesh
pip install tensorboard
pip install tqdm
pip install pyhocon
pip install icecream
pip install scipy
pip install PyMCubes==0.1.6
```

dvd(2024/10/11): For cluster
```shell
qlogin -q gpu
module load python # Check which version with module avail python
python -m venv .venv
source /work/imvia/de1450bo/repos/RNb-NeuS-fork/.venv/bin/activate

/work/imvia/de1450bo/repos/RNb-NeuS-fork/.venv/bin/python -m pip install --upgrade pip
/work/imvia/de1450bo/repos/RNb-NeuS-fork/.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
/work/imvia/de1450bo/repos/RNb-NeuS-fork/.venv/bin/python -m pip install opencv-python-headless trimesh tensorboard tqdm pyhocon icecream scipy PyMCubes==0.1.6

# exp_runner.py example:
/work/imvia/de1450bo/repos/RNb-NeuS-fork/.venv/bin/python exp_runner.py --mode train_rnb --conf /work/imvia/de1450bo/repos/RNb-NeuS-fork/confs/wmask_rnb.conf --case bearPNG_002
```

## Usage

dvd(2024/09/28): Python version: 3.9.13
```shell

```

#### Data Convention

Our data format is inspired from [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) as follows:
```
CASE_NAME
|-- cameras.npz    # camera parameters
|-- normal
    |-- 000.png        # normal map for each view
    |-- 001.png
    ...
|-- albedo
    |-- 000.png        # albedo for each view (optional)
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # mask for each view
    |-- 001.png
    ...
```

One can create folders with different data in it, for instance, a normal folder for each normal estimation method.
The name of the folder must be set in the used `.conf` file.

We provide the [DiLiGenT-MV](https://drive.google.com/file/d/1TEBM6Dd7IwjRqJX0p8JwT9hLmy_vA5nU/view?usp=drive_link) data as described above with normals and reflectance maps estimated with [SDM-UniPS](https://github.com/satoshi-ikehata/SDM-UniPS-CVPR2023/). Note that the reflectance maps were scaled over all views and uncertainty masks were generated from 100 normals estimations (see the article for further details).

### Run RNb-NeuS!

**Train with reflectance**

```shell
python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME
```

**Train without reflectance**

```shell
python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME --no_albedo
```

**Extract surface** 

```shell
python exp_runner.py --mode validate_mesh --conf ./confs/CONF_NAME.conf --case CASE_NAME --is_continue
```

Additionaly, we provide the five meshes of the DiLiGenT-MV dataset with our method [here](https://drive.google.com/file/d/1CTQW1YLWOT2sSEWznFmSY_cUUtiTXLdM/view?usp=drive_link).

## Citation
If you find our code useful for your research, please cite
```
@inproceedings{Brument23,
    title={RNb-Neus: Reflectance and normal Based reconstruction with NeuS},
    author={Baptiste Brument and Robin Bruneau and Yvain Quéau and Jean Mélou and François Lauze and Jean-Denis Durou and Lilian Calvet},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```
