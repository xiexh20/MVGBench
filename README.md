# MVGBench
A comprehensive benchmark suite for multi-view generation models

[Project Page](https://virtualhumans.mpi-inf.mpg.de/MVGBench/) | [ArXiv paper](https://virtualhumans.mpi-inf.mpg.de/MVGBench/MVGBench.pdf)

## Installation

The environment installation is similar to [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) environment. 

```shell
conda create -n mvgbench python=3.10 -y
conda activate mvgbench
pip install trimesh numpy==1.24.3 opencv-python==4.10.0.84 plyfile tqdm pillow==10.2.0 scikit-learn scikit-image lpips
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# compile 3dgs dependencies 
cd submodules
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
```

## Evaluation 
### 3D consistency metric 

Download example data

#### Synthetic images
Run 3dgs fitting
```shell
# 3DGS fitting 
python run_mvfit.py "example/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11v4_*/*" --white_background

# evaluation
python eval/eval_consistency.py \
  --name_odd output/consistency/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11v4_odd \
  --name_even output/consistency/syncdreamer+mvdfusion-v16-elev030-amb1.0+i000+sel11v4_even
```


#### Real images
Similar to synthetic images, we first need to do 3DGS fitting and then compute self consistency. 
We additionally need to perform an alignment on the optimized 3DGS to avoid biased metrics. 
The alignment takes output of one method as reference and aligns all others w.r.t one 3DGS fitting. 

```shell
python run_mvfit.py "example/*co3d2seq*/*" --white_background

python eval/align_3dgs.py --folder_tgt output/consistency/sv3dp+co3d2seq-sv3d-v21-manual+i000_even \
  --folder_src "output/consistency/${exp}+${rn}*${cname}*even"

# re-render with the alignment parameters

# evaluate

```
