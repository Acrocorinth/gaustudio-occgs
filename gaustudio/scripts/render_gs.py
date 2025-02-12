import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import trimesh
from gaustudio.cameras import *
from tqdm import tqdm


def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='vanilla')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--model', '-m', default=None, help='path to the model')
    parser.add_argument('--source_path', '-s', help='path to the dataset')
    parser.add_argument('--flythrough', action='store_true', help='render a flythrough path')
    parser.add_argument('--output-dir', '-o', default=None, help='path to the output dir')
    parser.add_argument('--load_iteration', default=-1, type=int, help='iteration to be rendered')
    parser.add_argument('--resolution', default=2, type=int, help='downscale resolution')
    parser.add_argument('--white_background', action='store_true', help='use white background')
    args, extras = parser.parse_known_args()
    
    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    from gaustudio import datasets, models, renderers
    from gaustudio.utils.misc import load_config
    # parse YAML config to OmegaConf
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '../configs', args.config+'.yaml')
    config = load_config(config_path, cli_args=extras)
    config.cmd_args = vars(args)  
    
    pcd = models.make(config.model.pointcloud)
    renderer = renderers.make(config.renderer)
    
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
    
    if args.source_path is None:
        args.source_path = os.path.join(os.path.dirname(model_path), "cameras.json")

    if args.source_path.endswith(".json"):
        print("Loading camera data from {}".format(args.source_path))
        with open(args.source_path, 'r') as f:
            camera_data = json.load(f)
        cameras = []
        for camera_json in camera_data:
            camera = JSON_to_camera(camera_json, "cuda")
            cameras.append(camera)
    else:
        dataset_config = { "name":"colmap", "source_path": args.source_path, 
                          "images":"images", "resolution":-1, "data_device":"cuda", "eval": False}
        dataset = datasets.make(dataset_config)
        cameras = dataset.all_cameras

    if args.flythrough:
        # Validate camera paths and optionally discard outliers
        window_size_ratio = 0.1
        speed_tolerance = 0.1
        discard_outliers = True
        cameras, invalid_cameras = validate_paths(cameras, window_size_ratio=window_size_ratio,
                                                speed_tolerance=speed_tolerance,
                                                discard_outliers=discard_outliers)

        # Downsample cameras with a minimum sample constraint
        translation_threshold = 0.1
        rotation_threshold = 5
        min_samples = 10
        cameras = downsample_cameras(cameras, translation_threshold=translation_threshold,
                                    rotation_threshold=rotation_threshold,
                                    min_samples=min_samples)

        # Smoothen camera paths with multiple iterations
        window_size_ratio = 1
        cameras = smoothen_cameras(cameras, window_size_ratio=window_size_ratio)

        # Upsample camera paths based on desired velocity
        meters_per_frame = 0.01
        angles_per_frame = 1
        cameras = upsample_cameras_velocity(cameras, meters_per_frame=meters_per_frame,
                                            angles_per_frame=angles_per_frame)

    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_path = os.path.join(work_dir, "images")
    mask_path = os.path.join(work_dir, "masks")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    rendered_images = []
    for camera in tqdm(cameras):
        camera.image = None
        camera.downsample_scale(args.resolution)
        camera = camera.to("cuda")
        with torch.no_grad():
            render_pkg = renderer.render(camera, pcd)
        rendering = render_pkg["render"]
        invalid_mask = render_pkg["rendered_final_opacity"][0] < 0.5
        rendering[:, invalid_mask] = 0.

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{camera.image_name}.png"))
        torchvision.utils.save_image((~invalid_mask).float(), os.path.join(mask_path, f"{camera.image_name}.png"))
        
        # Convert the rendered image to a numpy array
        rendering_np = rendering.permute(1, 2, 0).cpu().numpy() * 255
        rendering_np = rendering_np.astype(np.uint8)
        # Append the rendered image to the list
        rendered_images.append(rendering_np)

    # Create a video from the list of rendered images
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(rendered_images, fps=30)
    clip.write_videofile(os.path.join(work_dir, 'rgb.mp4'))
if __name__ == '__main__':
    main()