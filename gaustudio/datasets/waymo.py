import json
import math
import os
import pickle
import sys
import tempfile
from glob import glob
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gaustudio import datasets
from gaustudio.datasets.utils import focal2fov, getNerfppNorm

cameras = ["camera_FRONT", "camera_FRONT_LEFT", "camera_FRONT_RIGHT", "camera_SIDE_LEFT", "camera_SIDE_RIGHT"]
image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]
_label2camera = {
    0: "FRONT",
    1: "FRONT_LEFT",
    2: "FRONT_RIGHT",
    3: "SIDE_LEFT",
    4: "SIDE_RIGHT",
}


def image_filename_to_cam(x):
    return int(x.split(".")[0][-1])


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def image_filename_to_frame(x):
    return int(x.split(".")[0][:6])


def camera_to_JSON(id, camera: datasets.Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.image_width,
        "height": camera.image_height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FoVy, camera.image_height),
        "fx": fov2focal(camera.FoVx, camera.image_width),
        "cy": camera.image_height * camera.principal_point_ndc[1],
        "cx": camera.image_width * camera.principal_point_ndc[0],
    }
    return camera_entry


def generate_sample(scenario_data, frame_idx, camera_id):
    print(scenario_data["observers"].keys())
    camera_data = scenario_data["observers"][camera_id]
    n_frames = camera_data["n_frames"]
    if frame_idx >= n_frames:
        raise ValueError(f"Frame index {frame_idx} out of range for camera {camera_id} with {n_frames} frames.")

    # Load image
    image_path = f"images/{camera_id}/{frame_idx:08d}.jpg"
    image = np.array(Image.open(image_path))

    # Get frame metadata
    frame_json = camera_data["data"]
    for key in ["hw", "intr", "c2w", "distortion"]:
        frame_json[key] = frame_json[key][frame_idx]

    width, height = frame_json["hw"]
    fx, fy, cx, cy = (
        frame_json["intr"][0, 0],
        frame_json["intr"][1, 1],
        frame_json["intr"][0, 2],
        frame_json["intr"][1, 2],
    )

    intrinsic_4x4 = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    intrinsics = {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    c2w = frame_json["c2w"]

    sample = {"color": image, "c2w": c2w, "intrinsic_dict": intrinsics, "intrinsic_4x4": intrinsic_4x4}

    return sample


class WaymoDatasetBase:
    # the data only has to be processed once
    def __init__(self, config: Dict):
        self._validate_config(config)
        self.path = Path(config["source_path"])
        # scenario_path = self.path / "scenario.pt"
        self.camera_number = config.get("camera_number", 1)
        self.camera_ids = cameras[: self.camera_number]
        self.eval = config.get("eval", False)
        # with open(scenario_path, 'rb') as f:
        #     scenario_data = pickle.load(f)
        self._initialize(self.path)

    def _validate_config(self, config: Dict):
        required_keys = ["source_path"]
        for k in required_keys:
            if k not in config:
                raise ValueError(f"Config must contain '{k}' key")

    def downsample_scale(self, resolution_scale):
        self.all_cameras = [c.downsample_scale(resolution_scale) for c in self.all_cameras]

    def load_mask(self, datadir):
        sky_mask_dir = os.path.join(datadir, "sky_mask")
        dynamic_mask_dir = os.path.join(datadir, "dynamic_mask")
        test_mask_dir = os.path.join(datadir, "test_mask")
        os.makedirs(test_mask_dir, exist_ok=True)
        sky_mask_filenames_all = sorted(glob(os.path.join(sky_mask_dir, "*.jpg")))
        masks = [[] for i in range(5)]
        for filename in tqdm(sky_mask_filenames_all):
            image_basename = os.path.basename(filename)
            cam = image_filename_to_cam(filename)
            test_filename = os.path.join(test_mask_dir, image_basename)
            dynamic_mask_filename = os.path.join(dynamic_mask_dir, image_basename.replace(".jpg", ".png"))
            sky_mask = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            dynamic_mask = cv2.imread(str(dynamic_mask_filename), cv2.IMREAD_GRAYSCALE)
            sky_mask = np.array(sky_mask)
            dynamic_mask = np.array(dynamic_mask)
            mask = np.logical_or(sky_mask == 255, dynamic_mask == 255).astype(np.uint8)
            mask = np.where(mask, 0, 1).astype(np.uint8)
            # Image.fromarray(sky_mask).save(test_filename)
            mask = mask.transpose(0, 1)
            mask = torch.from_numpy(mask)
            masks[cam].append(mask)

        return masks

    def load_camera_info(self, datadir):
        ego_pose_dir = os.path.join(datadir, "ego_pose")
        extrinsics_dir = os.path.join(datadir, "extrinsics")
        intrinsics_dir = os.path.join(datadir, "intrinsics")

        intrinsics = []
        extrinsics = []
        for i in range(5):
            intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            intrinsics.append(intrinsic)

        for i in range(5):
            cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
            extrinsics.append(cam_to_ego)

        ego_frame_poses = []
        ego_cam_poses = [[] for i in range(5)]
        ego_pose_paths = sorted(os.listdir(ego_pose_dir))
        for ego_pose_path in ego_pose_paths:
            # frame pose
            if "_" not in ego_pose_path:
                ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
                ego_frame_poses.append(ego_frame_pose)
            else:
                cam = image_filename_to_cam(ego_pose_path)
                ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
                ego_cam_poses[cam].append(ego_cam_pose)

        # # center ego pose
        ego_frame_poses = np.array(ego_frame_poses)
        center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
        ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

        ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
        ego_cam_poses = np.array(ego_cam_poses)
        ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
        return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses

    def _initialize(self, cam_info_path):
        all_cameras_unsorted = []
        intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = self.load_camera_info(cam_info_path)
        masks = self.load_mask(cam_info_path)
        scenario_path = os.path.join(cam_info_path, "scenario.pt")
        with open(scenario_path, "rb") as f:
            scenario_data = pickle.load(f)
        image_dir = os.path.join(cam_info_path, "images")
        undistort_dir = os.path.join(cam_info_path, "undistorted_images")
        os.makedirs(undistort_dir, exist_ok=True)
        undistort_intr_dir = os.path.join(cam_info_path, "undistorted_intrinsics")
        os.makedirs(undistort_intr_dir, exist_ok=True)
        image_filenames_all = sorted(glob(os.path.join(image_dir, "*.jpg")))
        for filename in tqdm(image_filenames_all):
            image_basename = os.path.basename(filename)
            frame = image_filename_to_frame(image_basename)
            cam = image_filename_to_cam(image_basename)
            camera_data = scenario_data["observers"][cameras[cam]]
            frame_json = camera_data["data"]
            distortion_coeffs = frame_json["distortion"][frame]
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            mask = masks[cam][frame]
            c2w = pose @ ext

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            h, w = image_heights[cam], image_widths[cam]

            img = cv2.imread(str(filename))
            undistort_intr, roi = cv2.getOptimalNewCameraMatrix(ixt, distortion_coeffs, (w, h), 0)
            np.savetxt(os.path.join(undistort_intr_dir, image_basename.replace(".jpg", ".txt")), undistort_intr)
            undistorted_img = cv2.undistort(img, ixt, distortion_coeffs, None, undistort_intr)
            undistort_filename = os.path.join(undistort_dir, image_basename)
            cv2.imwrite(str(undistort_filename), undistorted_img)
            fx, fy, cx, cy = undistort_intr[0, 0], undistort_intr[1, 1], undistort_intr[0, 2], undistort_intr[1, 2]
            FoVy = focal2fov(fy, h)
            FoVx = focal2fov(fx, w)

            _camera = datasets.Camera(
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                image_path=undistort_filename,
                image_name=image_basename,
                image_width=w,
                image_height=h,
                mask=mask,
                principal_point_ndc=np.array([cx / w, cy / h]),
            )
            all_cameras_unsorted.append(_camera)
            # Get frame metadata
        self.all_cameras = sorted(all_cameras_unsorted, key=lambda x: x.image_name)
        self.nerf_normalization = getNerfppNorm(self.all_cameras)
        self.cameras_extent = self.nerf_normalization["radius"]

    def export(self, save_path):
        json_cams = []
        camlist = []
        camlist.extend(self.all_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(save_path, 'w') as file:
            json.dump(json_cams, file)

@datasets.register('waymo')
class WaymoDataset(Dataset, WaymoDatasetBase):
    def __init__(self, config):
        super().__init__(config)

    def __len__(self):
        return len(self.all_cameras)

    def __getitem__(self, index):
        return self.all_cameras[index]
