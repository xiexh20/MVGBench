# Evaluating your own MVG model

### Step 1: render input
For best setup performance, you should render the input images using the camera setup used during training time. 
We provide most commonly used setup from SV3D, SyncDreamer, Zero123 and V3D, see our supplementary table 5 for the exact setups. 

These renderings can be downloaded from [here](), the files are organized as:
```shell
|--gso100 # for 100 GSO objects
|----object 
|------<render_name>
|--------<000-0xx.png> # multi-view rendering, 000.png is the front/input view 
|--------transforms.json # the camera pose parameters 
```

We also provide a script to render input images using blender, example command:
```shell
{blender_path} -b -P render/blender_render.py -- {mesh_obj_file} --out_name {out_name} --views {n_views} \
            --camera_type {camera_type} --world_color {world_color} --resolution 1024  --engine CYCLES   
```
Details for the input arguments:
- `blender_path`: blender executable file, we used blender 3.5 in this project. 
- `mesh_obj_file`: the file path to GSO or Omniobject3D objects. e.g. `root/alarm/meshes/model.obj` for GSO and `root/squash_003/Scan/Scan.obj` for OmniObj3D.
- `out_name`: for convenience, the convention we use for `out_name` is `gso100-{method}-v{n_views}-elev{int(elev):03d}-amb{world_color}`.
- `n_views`: how many views does your model generate. 
- `camera_type`: the name of your method/camera type, this is used to setup camera focal, distance, and elevation. You should add your own camera setup in [render/blender_render.py](../render/blender_render.py#L209-L290).
- `world_color`: the ambience light intensity, default should be 1.0.  
This script will save the rendered images and a `transforms.json` file storing camera parameters which will be used later for 3DGS fitting.  

### Step 2: run MVG generation
Given the input images from step 1, you can run your own MVG model to generate multi-view images. 

### Step 3: format generated images into 3DGS fitting format
View splits: please check paper supplementary section 6.1 for splitting the generated images. 
See this [example script](https://github.com/xiexh20/ViFiGen/blob/main/scripts/full_pipeline.py) about how to format generated multi-view images, i.e. method `format_output`

### Step 4: run evaluation
Please refer to the [main doc](../README.md#evaluation-) for evaluation.





