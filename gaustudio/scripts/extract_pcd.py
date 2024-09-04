import sys
import argparse
import os
import time
import logging
from datetime import datetime
import torch
import json
from pathlib import Path
import cv2
import torchvision
from tqdm import tqdm
import trimesh
import numpy as np
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='vanilla')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--camera', '-c', default=None, help='path to cameras.json')
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--output-dir', '-o', default=None, help='path to the output dir')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--resolution', default=2, type=int, help='downscale resolution')
    parser.add_argument('--sh', default=0, type=int, help='default SH degree')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    parser.add_argument('--clean', action='store_true', help='perform a clean operation')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))
    
    from gaustudio.utils.misc import load_config
    from gaustudio import models, renderers
    from gaustudio.utils.cameras_utils import JSON_to_camera
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', args.config+'.yaml')
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)  
    
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    pcd.active_sh_degree = args.sh
    
    model_path = args.model
    if os.path.isdir(model_path):
        if args.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(args.model, "point_cloud"))
        else:
            loaded_iter = args.load_iteration
        work_dir = os.path.join(model_path, "renders", "iteration_{}".format(loaded_iter)) if args.output_dir is None else args.output_dir
        
        print("Loading trained model at iteration {}".format(loaded_iter))
        pcd.load(os.path.join(args.model,"point_cloud", "iteration_" + str(loaded_iter), "point_cloud.ply"))
    elif model_path.endswith(".ply"):
        work_dir = os.path.join(os.path.dirname(model_path), os.path.basename(model_path)[:-4]) if args.output_dir is None else args.output_dir
        pcd.load(model_path)
    else:
        print("Model not found at {}".format(model_path))
    pcd.to("cuda")
    
    if args.camera is None:
        args.camera = os.path.join(model_path, "cameras.json")
    if os.path.exists(args.camera):
        print("Loading camera data from {}".format(args.camera))
        with open(args.camera, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        assert "Camera data not found at {}".format(args.camera)
    
    from gaustudio.utils.sh_utils import SH2RGB
    from gaustudio.datasets.utils import getNerfppNorm
    scene_radius = getNerfppNorm(cameras)["radius"]
    all_ids = []
    all_normals = []
    for camera in tqdm(cameras[::3]):
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendered_final_opacity =  render_pkg["rendered_final_opacity"][0]
        rendered_depth = render_pkg["rendered_depth"][0]
        normals = camera.depth2normal(rendered_depth, coordinate='world')        
        median_point_depths =  render_pkg["rendered_median_depth"][0]
        median_point_ids =  render_pkg["rendered_median_depth"][2].int()
        median_point_weights =  render_pkg["rendered_median_depth"][1]
        valid_mask = (rendered_final_opacity > 0.5) & (median_point_weights > 0.1)
        valid_mask = (median_point_depths < scene_radius * 1.5) & valid_mask
        valid_mask = (normals.sum(dim=-1) > -3) & valid_mask

        median_point_ids = median_point_ids[valid_mask]
        median_point_normals = -normals[valid_mask]
        
        all_ids.append(median_point_ids)
        all_normals.append(median_point_normals)
    all_ids = torch.cat(all_ids, dim=0)
    all_normals = torch.cat(all_normals, dim=0)
    
    # fusion
    unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
    num_unique_ids = len(unique_ids)
    sum_normals = torch.zeros((num_unique_ids, all_normals.size(1)), device=all_normals.device)
    counts = torch.zeros(num_unique_ids, device=all_ids.device)
    sum_normals.index_add_(0, inverse_indices, all_normals)
    counts.index_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
    mean_normals = sum_normals / counts.unsqueeze(1)

    import open3d as o3d
    surface_xyz_np = pcd._xyz[unique_ids].cpu().numpy()
    surface_color_np = SH2RGB(pcd._f_dc[unique_ids]).clip(0,1).cpu().numpy()
    surface_normal_np = mean_normals.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(surface_xyz_np)
    pcd.colors = o3d.utility.Vector3dVector(surface_color_np)
    pcd.normals = o3d.utility.Vector3dVector(surface_normal_np)
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "fused.ply"), pcd)
    
    import nksr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_xyz = torch.from_numpy(surface_xyz_np).float().to(device)
    input_normal = torch.from_numpy(surface_normal_np).float().to(device)

    # Perform reconstruction
    reconstructor = nksr.Reconstructor(device)
    field = reconstructor.reconstruct(input_xyz, input_normal, detail_level=0.5, voxel_size=0.04)
    mesh = field.extract_dual_mesh(mise_iter=1)
    mesh = trimesh.Trimesh(vertices=mesh.v.cpu().numpy(), faces=mesh.f.cpu().numpy())
    mesh.export(os.path.join(args.output_dir, "fused_mesh.ply"))

if __name__ == '__main__':
    main()