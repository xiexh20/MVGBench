"""
a base class for GSO rendering using blender
"""
import argparse, sys, os, math, re
from datetime import datetime

import bpy
import numpy as np
from glob import glob
import os.path as osp
import json, random
from mathutils import Vector

SAVE_ALBEDO = False
SAVE_DEPTH = False
SAVE_NORMAL = False


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    # TODO: understand if this is really uniform sample
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3) # this is not true uniform sampling on the sphere!
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]

        # compute elevation angle
        elev = np.rad2deg(np.arcsin(vec[2]/radius[0]))

        if maxz > vec[2] > minz and 45 >= elev >= -15:
            print('Final elevation:', elev)
            correct = True
    return vec

def uniform_elev(radius_min=1.5, radius_max=2.0):
    "sample uniform at elevation angle"
    phi = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
    theta = np.random.uniform(45, 105.)
    theta = np.deg2rad(theta)
    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    radius = np.random.uniform(radius_min, radius_max)
    return np.array([x, y, z]) * radius


def randomize_camera():
    # from https://blender.stackexchange.com/questions/18530/
    radius_max, radius_min = 3.8, 3.0 # random-v16-v4 setup
    # random-v16-fov45-d2.8-3.5 setup:
    radius_max, radius_min = 2.8, 3.5
    # x, y, z = sample_spherical(radius_min=radius_min, radius_max=radius_max, maxz=radius_max, minz=-radius_max)
    x, y, z = uniform_elev(radius_min=radius_min, radius_max=radius_max) # random-v16-v4
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera


class ImageRenderer:
    def __init__(self, args):
        "setup blender"
        # Set up rendering
        # bpy.ops.wm.open_mainfile(filepath='/home/ubuntu/efs/static/cube-area-lights.blend')
        # bpy.ops.wm.open_mainfile(filepath='/home/ubuntu/efs/static/background_bright-nocamera.blend') # bright rendering
        # save image as sRGB and white background blend/sRGB-white-nolights.blend
        bpy.ops.wm.open_mainfile(filepath='/home/ubuntu/efs/static/blend/sRGB-white-nolights.blend')
        bpy.ops.object.camera_add(location=(0, 0, 0))  # add a camera and set it as the scene camera
        bpy.context.scene.camera = bpy.context.object

        # set up ambient color density
        bpy.context.scene.world.color = (args.world_color, ) * 3

        # Iterate over all lamps in the scene
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                # Disable shadow
                obj.data.use_shadow = False

        scene = bpy.context.scene
        render = bpy.context.scene.render

        render.engine = args.engine
        # Set device to GPU, copy Objaverse setup
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.samples = 128
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        # Set the device_type
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = args.device  # or "OPENCL"
        bpy.context.scene.cycles.tile_size = 8192
        # setup which gpu to use, or don't need to setup gpu id, the CUDA_VISIBLE_DEVICE also works!
        # cuda_devices = bpy.context.preferences.addons["cycles"].preferences.get_devices_for_type('CUDA')
        # this seems to be the key to use gpu for rendering
        bpy.context.preferences.addons["cycles"].preferences.get_devices_for_type('CUDA')[args.gpu_id].use = True

        # Get list of CUDA devices
        # devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()
        # for blender 3.0, see: https://devtalk.blender.org/t/blender-v2-82-vs-v3-0-1-gpu-is-semi-detected-aws-server-nvidia-a10g-gpu/22805
        # devices = bpy.context.preferences.addons['cycles'].preferences.devices
        # import pdb;pdb.set_trace()
        # Enable all CUDA devices
        # for device in devices:
        #     print(device.name, device.type)
        #     if device.type == 'CUDA':
        #         device.use = True
        #         print('Enabled one GPU')  # TODO: understand why this does not work! why no GPU/CPU in a10g!


        render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
        render.image_settings.color_depth = args.color_depth  # ('8', '16')
        render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
        render.resolution_x = args.resolution
        render.resolution_y = args.resolution
        render.resolution_percentage = 100
        render.film_transparent = True

        scene.use_nodes = True
        scene.view_layers["ViewLayer"].use_pass_normal = True
        scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
        scene.view_layers["ViewLayer"].use_pass_object_index = True

        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        # Clear default nodes
        for n in nodes:
            nodes.remove(n)

        # Create input render layer node
        render_layers = nodes.new('CompositorNodeRLayers')

        # Create depth output nodes
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = args.format
        depth_file_output.format.color_depth = args.color_depth
        if args.format == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            depth_file_output.format.color_mode = "BW"

            # Remap as other types can not represent the full range of depth.
            map = nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.offset = [-0.7]
            map.size = [args.depth_scale]
            map.use_min = True
            map.min = [0]

            links.new(render_layers.outputs['Depth'], map.inputs[0])
            links.new(map.outputs[0], depth_file_output.inputs[0])

        # Create normal output nodes
        scale_node = nodes.new(type="CompositorNodeMixRGB")
        scale_node.blend_type = 'MULTIPLY'
        # scale_node.use_alpha = True
        scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_node.inputs[1])

        bias_node = nodes.new(type="CompositorNodeMixRGB")
        bias_node.blend_type = 'ADD'
        # bias_node.use_alpha = True
        bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_node.outputs[0], bias_node.inputs[1])

        normal_file_output = nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = args.format
        links.new(bias_node.outputs[0], normal_file_output.inputs[0])

        # Create albedo output nodes
        if SAVE_ALBEDO:
            alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
            links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
            links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

            albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
            albedo_file_output.label = 'Albedo Output'
            albedo_file_output.base_path = ''
            albedo_file_output.file_slots[0].use_node_format = True
            albedo_file_output.format.file_format = args.format
            albedo_file_output.format.color_mode = 'RGBA'
            albedo_file_output.format.color_depth = args.color_depth
            links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
            self.albedo_file_output = albedo_file_output
        self.normal_file_output = normal_file_output
        self.depth_file_output = depth_file_output



    def render(self, args):
        context = bpy.context
        scene = bpy.context.scene
        # output path
        # path for GSO dataset structure
        outfolder = osp.join(osp.dirname(osp.dirname(args.obj)), args.out_name)

        # check if it is done
        files = glob(outfolder+"/*.png")
        if len(files) == args.views and not args.redo and os.path.isfile(outfolder+"/transforms.json"):
            if not args.only_first:
                # check if there is synlink
                done = True
                for file in files:
                    if osp.islink(file):
                        print('find synlink in {}'.format(file))
                        os.remove(file)
                        done = False
            if done:
                print(outfolder, 'already done, skipped')
                return

        # Import textured mesh
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_scene.obj(filepath=args.obj, axis_forward='Y', axis_up='Z')  # prevent any rotation by blender
        obj = bpy.context.selected_objects[0]
        if args.is_omni3d:
            # align the coordinate with GSO: xyz Euler mode
            obj.rotation_euler[0] = np.deg2rad(90)
            obj.rotation_euler[2] = np.deg2rad(-90)

        bpy.context.view_layer.objects.active = obj

        os.makedirs(outfolder, exist_ok=True)
        fp = osp.join(outfolder, '')

        # normalize the object
        obj_size, vcen, verts = self.calc_norm_params(obj)

        # Possibly disable specular shading
        for slot in obj.material_slots:
            node = slot.material.node_tree.nodes['Principled BSDF']
            node.inputs['Specular'].default_value = 0.05

        if args.scale != 1:
            bpy.ops.transform.resize(value=(args.scale, args.scale, args.scale))
            bpy.ops.object.transform_apply(scale=True)
        if args.remove_doubles:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.remove_doubles()
            bpy.ops.object.mode_set(mode='OBJECT')
        if args.edge_split:
            bpy.ops.object.modifier_add(type='EDGE_SPLIT')
            context.object.modifiers["EdgeSplit"].split_angle = 1.32645
            bpy.ops.object.modifier_apply(modifier="EdgeSplit")

        # Set objekt IDs
        obj.pass_index = 1

        # Place camera
        cam = scene.objects['Camera']
        # cam.location = (0, 1, 0.6)
        # lens, sensor_width, angle are the same, changing one and the others will be updated automatically
        # cam.data.lens = 35
        # cam.data.sensor_width = 32
        # cam.data.angle = np.deg2rad(33.8)

        # setup all cameras with same sensor width
        cam.data.lens = 35
        cam.data.sensor_width = 32

        # for lemon, the normalization is somehow not done properly
        # normalize the object
        obj.scale *= 1.0 / obj_size # normalize to -1, 1 for better 3DGS
        obj.location = 1.0 * (obj.location - Vector(vcen)) / obj_size  # the translation should also be updated, as it is s*p + loc
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        if 'lemon_005' in args.obj: # fix bug for lemon, do not affect others
            obj_size, vcen, verts = self.calc_norm_params(obj)
            assert obj_size == 1.0, 'something wrong in normalization!'
            obj.scale *= 1.0 / obj_size  # normalize to -1, 1 for better 3DGS
            obj.location = 1.0 * (obj.location - Vector(vcen)) / obj_size  # the translation should also be updated, as it is s*p + loc

        # setup camera intrinsic and elevation angle
        if args.camera_type == 'sv3d':
            # SV3D setup: elevation [-5, 30], compute a margin at around 0.7
            args.fov = 33.8
            # dist = 2.35, or 4.701982230456558
            dist = 2./(1.4 * np.tan(np.deg2rad(args.fov)/2)) # the object occupies roughly 0.7 of the image
            args.elev = 25/2. # 12.5
        elif args.camera_type == 'sv3dv2':
            # SV3D setup v2: elevation [-5, 30], compute a margin at around 0.8
            args.fov = 33.8
            # dist = 2.35, or 4.701982230456558
            dist = 2./(1.6 * np.tan(np.deg2rad(args.fov)/2)) # the object occupies roughly 0.8 of the image
            args.elev = 25/2.
        elif args.camera_type == 'zero123':
            # zero123 setup: random pose, random distance [1.5, 2.2], fov 49.1, object normalized to [-0.5, 0.5]
            args.elev = 0.0
            args.fov = 49.1
            dist = 2 * (1.5 + 2.2)/2. # 1.85m
        elif args.camera_type == 'stable-zero123':
            # stable-zero123, see: https://github.com/DSaurus/threestudio-mvimg-gen/tree/master?tab=readme-ov-file#camera-parameters-in-config-file
            args.elev = 0.0
            args.fov = 20.0
            dist = 3.8 * 2
        elif args.camera_type == 'v3d':
            # v3d setup: dist=2, elev=0, fov=60, N=18 views, see https://github.com/heheyas/V3D/blob/main/recon/arguments/__init__.py#L67
            args.elev = 0.0
            args.fov = 60.0
            # dist = 3.0 # 2.0 can lead to too over-crop
            # the mesh might be scaled based on maximum radius?
            verts = verts * 1.0/obj_size # apply the transformation
            vmin = verts.min(axis=0)
            vmax = verts.max(axis=0)
            vcen = (vmin + vmax) / 2
            obj_size = np.sqrt(np.max(np.sum((verts - vcen)**2, -1)))
            dist = 2.0 * obj_size
            print(f"Object size: {obj_size}, camera distance: {dist}")
            # return
        elif args.camera_type == 'others':
            dist = args.dist
            pass # use the parameters set from command line arguments
        elif args.camera_type == 'spad':
            # see SPAD section 4.1 training data curation
            args.elev, args.fov = 0., 40.26
            dist = 3.5
        elif args.camera_type == 'mvdfusion':
            # see https://github.com/zhizdev/mvdfusion/issues/10#issuecomment-2198421208
            # clamp issue?
            # also the saame for Hi3D
            args.fov = np.rad2deg(0.8575560450553894) # 49.13 degree
            args.elev = 30.
            dist = 3.0
            pass
        elif args.camera_type == 'random':
            # random views, the elevation is set to zero, but are randomized later
            args.elev = 0
            args.fov = 42 # would this camera distance favor sv3d?
            dist = 3.2 # will sample from 3.0 to 3.8
            # random-v16-fov45-d2.8-3.5 setup:
            args.fov = 45
        elif args.camera_type== 'syncdreamer':
            # epidiff, syncdreamer seem to have similar input rendering setup
            # zero123++: unknown yet.
            # wonder3d: might have a different camera distance
            args.fov = cam.data.angle # use the one set by lens and sensor_width, same as zero123
            args.elev = 15. # random elevation between -10 and 40
            dist = 3.0 # same as the mvdfusion
        elif args.camera_type == 'random-focal':
            # random focal to test the case for real images
            args.fov = np.random.uniform(33., 50)
            args.elev = 0
            dist = np.random.uniform(3.7, 4.7)
        else:
            raise ValueError('Unknown camera type')
        cam.data.angle = np.deg2rad(args.fov)

        if args.force_elev is not None:
            args.elev = args.force_elev
            print(f"Force elevation angle to be {args.elev}")

        # the world coordinate: x-right, y-forward, z-up
        # azimuth 0 corresponds to (0, -1, z)
        cam.location = (dist*np.sin(0.)* np.cos(np.deg2rad(args.elev)),
                        -dist * np.cos(np.deg2rad(args.elev))*np.cos(0.),
                        np.sin(np.deg2rad(args.elev)) * dist)

        if args.camera_type != 'random':
            # set target camera and track, for random camera, the extrinsics are randomly sampled
            cam_constraint = cam.constraints.new(type='TRACK_TO')
            cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
            cam_constraint.up_axis = 'UP_Y'
            cam_empty = bpy.data.objects.new("Empty", None)
            cam_empty.location = (0, 0, 0)
            cam.parent = cam_empty

            scene.collection.objects.link(cam_empty)
            context.view_layer.objects.active = cam_empty
            cam_constraint.target = cam_empty
            # specify a start azimuth
            cam_empty.rotation_euler[2] = args.start_azimuth
        else:
            randomize_camera()

        stepsize = 360.0 / args.views
        # offset = 7.0 # azimuth offset
        # cam_empty.rotation_euler[2] = math.radians(offset) # azimuth offset
        # assert '09-10' in str(datetime.now())
        rotation_mode = 'XYZ'

        # bpy.ops.wm.save_as_mainfile(filepath=osp.join(outfolder, 'after_track.blend'))

        out_data = {
            'obj_path': args.obj,
            "camera_angle_x": cam.data.angle,  # use the NeRF format
            'object_size': obj_size,
            'obj_center': [x for x in vcen],  # convert to list
            'elevation': np.deg2rad(args.elev),
            'distance': dist

        }
        out_data['frames'] = []
        for i in range(0, args.views):
            print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

            # render_file_path = fp + '_r_{0:03d}'.format(int(i * stepsize))
            render_file_path = fp + f'{i:03d}'

            scene.render.filepath = render_file_path
            if SAVE_DEPTH:
                self.depth_file_output.file_slots[0].path = render_file_path + "_depth"
            if SAVE_NORMAL:
                self.normal_file_output.file_slots[0].path = render_file_path + "_normal"
            if SAVE_ALBEDO:
                self.albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
            # id_file_output.file_slots[0].path = render_file_path + "_id"

            if i != 0 and args.only_first:
                # render only first view, others are synlinks
                os.system(f'ln -s {fp}000.png {render_file_path}.png')
                bpy.context.view_layer.update() # simulate rendering to update matrix_world, debug passed!
            else:
                bpy.ops.render.render(write_still=True)  # render still

            # save camera transform
            frame_data = {
                'file_path': f'./{osp.basename(render_file_path)}',
                'transform_matrix': listify_matrix(cam.matrix_world),  # this is the camera to world transform

            }
            out_data['frames'].append(frame_data)

            # update camera for next round, no difference whether this happens before or after??
            if args.camera_type != 'random':
                cam_empty.rotation_euler[2] += math.radians(stepsize)  # rotate the object in origin, and then the rendering camera will follow automatically
            else:
                randomize_camera()

        with open(outfolder + '/' + 'transforms.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=2)

    def calc_norm_params(self, obj):
        verts = np.array([tuple(obj.matrix_world @ v.co) for v in obj.data.vertices])
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        vcen = (vmin + vmax) / 2
        obj_size = np.abs(verts - vcen).max()
        return obj_size, vcen, verts

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
        parser.add_argument('obj', type=str,
                            help='Path to the obj file to be rendered.')
        parser.add_argument('--views', type=int, default=21,
                            help='number of views to be rendered')
        parser.add_argument('--output_folder', type=str, default='debug',
                            help='The path the output will be dumped to.')
        parser.add_argument('--scale', type=float, default=1,
                            help='Scaling factor applied to model. Depends on size of mesh.')
        parser.add_argument('--remove_doubles', type=bool, default=True,
                            help='Remove double vertices to improve mesh quality.')
        parser.add_argument('--edge_split', type=bool, default=True,
                            help='Adds edge split filter.')
        parser.add_argument('--depth_scale', type=float, default=1.4,
                            help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
        parser.add_argument('--color_depth', type=str, default='8',
                            help='Number of bit per channel used for output. Either 8 or 16.')
        parser.add_argument('--format', type=str, default='PNG',
                            help='Format of files generated. Either PNG or OPEN_EXR')
        parser.add_argument('--resolution', type=int, default=1024,
                            help='Resolution of the images.')
        parser.add_argument('--engine', type=str, default='CYCLES',
                            help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
        parser.add_argument("--device", type=str, default='CUDA')
        parser.add_argument('--gpu_id', default=0, type=int)
        parser.add_argument('-on', '--out_name', default='render', help='name of output folder')
        parser.add_argument('-elev', default=15, type=float, )
        parser.add_argument('-dist', default=1.6, type=float, )
        parser.add_argument('-fov', default=33.8, type=float, help='field of view in degree')
        parser.add_argument('-ct', '--camera_type', default='sv3d',
                            choices=['sv3d', 'zero123', 'imagedream', 'spad', 'mvdfusion', 'others',
                                     'random', 'random-focal', 'sv3dv2', 'stable-zero123', 'v3d'])
        parser.add_argument('-fe', '--force_elev', default=None, type=float)
        # parser.add_argument('--random', default=False, action='store_true', ) # use random camera pose or not
        parser.add_argument('--start_azimuth', default=0, type=float)
        parser.add_argument('--world_color', default=1.0, type=float, help='ambient color density')
        parser.add_argument('--redo', default=False, action='store_true')
        parser.add_argument('--only_first', default=False, action='store_true')

        parser.add_argument('--is_omni3d', default=False, action='store_true')

        return parser

def main():
    parser = ImageRenderer.get_parser()
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    renderer = ImageRenderer(args)
    renderer.render(args)

if __name__ == '__main__':
    main()
