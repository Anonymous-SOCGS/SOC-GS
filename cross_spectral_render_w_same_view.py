#!/bin/python3

import argparse
import taichi as ti
import torchvision.transforms as transforms
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch, quaternion_to_rotation_matrix_torch
from dataclasses import dataclass
from taichi_3d_gaussian_splatting.ImagePoseDataset import ImagePoseDataset

import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm
# from pytorch_msssim import ssim
import cv2
import time
import pynvml

TEST_ID = ['8', '10', '14', '20', '22']
GPU_ID = 0
pynvml.nvmlInit()
nvml_handler = pynvml.nvmlDeviceGetHandleByIndex(GPU_ID)
pynvml.nvmlInit()


def get_gpu_mem():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(nvml_handler)
    used_mem = round(meminfo.used / 1024 / 1024, 2)
    return used_mem


class GaussianPointRenderer:
    @dataclass
    class GaussianPointRendererConfig:
        parquet_path: str
        cameras: torch.Tensor
        cameras_ms: torch.Tensor
        cameras_ir: torch.Tensor
        device: str = "cuda"
        camera_id: int = 0
        image_height: int = 544
        image_width: int = 976
        image_height_ms: int = 544
        image_width_ms: int = 976
        image_height_ir: int = 544
        image_width_ir: int = 976
        camera_intrinsics: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 488.0], [0.0, 581.743, 272.0], [0.0, 0.0, 1.0]],
            device="cuda")
        camera_intrinsics_ms: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 488.0], [0.0, 581.743, 272.0], [0.0, 0.0, 1.0]],
            device="cuda")
        camera_intrinsics_ir: torch.Tensor = torch.tensor(
            [[581.743, 0.0, 488.0], [0.0, 581.743, 272.0], [0.0, 0.0, 1.0]],
            device="cuda")

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, config: GaussianPointRendererConfig) -> None:
        self.config = config
        self.ToPIL = transforms.ToPILImage()
        self.config.image_height = self.config.image_height - self.config.image_height % 16
        self.config.image_width = self.config.image_width - self.config.image_width % 16
        scene = GaussianPointCloudScene.from_trained_parquet(
            config.parquet_path, config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None))
        self.scene = self._merge_scenes([scene])
        with torch.no_grad():
            self.scene.point_cloud[torch.isnan(self.scene.point_cloud)] = 0
            self.scene.point_cloud_features[torch.isnan(self.scene.point_cloud_features)] = 0
        self.scene = self.scene.to(self.config.device)
        self.cameras = self.config.cameras.to(self.config.device)
        self.cameras_ms = self.config.cameras_ms.to(self.config.device)
        self.cameras_ir = self.config.cameras_ir.to(self.config.device)
        self.camera_info = CameraInfo(
            camera_intrinsics=self.config.camera_intrinsics.to(self.config.device),
            camera_width=self.config.image_width,
            camera_height=self.config.image_height,
            camera_intrinsics_multispectral=self.config.camera_intrinsics_ms,
            camera_height_multispectral=self.config.image_height_ms,
            camera_width_multispectral=self.config.image_width_ms,
            camera_intrinsics_infrared=self.config.camera_intrinsics_ir,
            camera_height_infrared=self.config.image_height_ir,
            camera_width_infrared=self.config.image_width_ir,
            camera_id=self.config.camera_id,
        )
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.4,
                far_plane=2000.,
                depth_to_sort_key_scale=10.))

    def _merge_scenes(self, scene_list):
        merged_point_cloud = torch.cat([scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = \
            torch.cat([scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = \
            [scene.point_cloud.shape[0] for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene_list[idx].point_cloud.mean(dim=0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = \
            torch.zeros((merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            ))
        return merged_scene

    def run(self, output_prefix, camera_modality='RGB'):
        save_dir_rgb = os.path.join(output_prefix, 'rgb')
        save_dir_ms = os.path.join(output_prefix, 'ms')
        save_dir_ir = os.path.join(output_prefix, 'ir')
        os.makedirs(save_dir_rgb, exist_ok=True)
        os.makedirs(save_dir_ms, exist_ok=True)
        os.makedirs(save_dir_ir, exist_ok=True)
        render_time = []
        render_mem_occupancy = []

        if camera_modality == 'rgb':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)
                with torch.no_grad():
                    start_time = time.time()
                    rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasteriser(
                        GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            camera_info=self.camera_info,
                            q_pointcloud_camera=q,
                            t_pointcloud_camera=t,
                            color_max_sh_band=3,
                        ),
                        current_train_stage='val_all',
                    )
                    end_time = time.time()
                    if i == 0:
                        pass
                    elif i > 0:
                        render_time.append(end_time - start_time)
                        render_mem_occupancy.append(get_gpu_mem())
                    rasterized_image_rgb = rasterized_image[:, :, :3]
                    rasterized_image_ms = rasterized_image[:, :, 3:4]
                    rasterized_image_ms = rasterized_image_ms.repeat(1, 1, 3)
                    rasterized_image_ir = rasterized_image[:, :, 4:]
                    rasterized_image_ir = rasterized_image_ir.repeat(1, 1, 3)

                    rasterized_image_rgb = torch.clamp(rasterized_image_rgb, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ms = torch.clamp(rasterized_image_ms, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ir = torch.clamp(rasterized_image_ir, min=0, max=1).permute(2, 0, 1)

                    image_rgb = self.ToPIL(rasterized_image_rgb)
                    image_ms = self.ToPIL(rasterized_image_ms)
                    image_ir = self.ToPIL(rasterized_image_ir)

                    image_rgb.save(os.path.join(save_dir_rgb, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ms.save(os.path.join(save_dir_ms, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ir.save(os.path.join(save_dir_ir, f'{TEST_ID[i].zfill(4)}.png'))
            print(f"Average Render Time:{sum(render_time) / len(render_time)}s")
            print(f"Average CUDA Memory Occupancy: {sum(render_mem_occupancy) / len(render_mem_occupancy)}MB")
        elif camera_modality == 'ms':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras_ms[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)
                with torch.no_grad():
                    start_time = time.time()
                    rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasteriser(
                        GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            camera_info=camera_info_ms,
                            q_pointcloud_camera=q,
                            t_pointcloud_camera=t,
                            color_max_sh_band=3,
                        ),
                        current_train_stage='val_all',
                    )
                    end_time = time.time()
                    if i == 0:
                        pass
                    elif i > 0:
                        render_time.append(end_time - start_time)
                        render_mem_occupancy.append(get_gpu_mem())
                    rasterized_image_rgb = rasterized_image[:, :, :3]
                    rasterized_image_ms = rasterized_image[:, :, 3:4]
                    rasterized_image_ms = rasterized_image_ms.repeat(1, 1, 3)
                    rasterized_image_ir = rasterized_image[:, :, 4:]
                    rasterized_image_ir = rasterized_image_ir.repeat(1, 1, 3)

                    rasterized_image_rgb = torch.clamp(rasterized_image_rgb, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ms = torch.clamp(rasterized_image_ms, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ir = torch.clamp(rasterized_image_ir, min=0, max=1).permute(2, 0, 1)

                    image_rgb = self.ToPIL(rasterized_image_rgb)
                    image_ms = self.ToPIL(rasterized_image_ms)
                    image_ir = self.ToPIL(rasterized_image_ir)

                    image_rgb.save(os.path.join(save_dir_rgb, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ms.save(os.path.join(save_dir_ms, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ir.save(os.path.join(save_dir_ir, f'{TEST_ID[i].zfill(4)}.png'))
            print(f"Average Render Time:{sum(render_time) / len(render_time)}s")
            print(f"Average CUDA Memory Occupancy: {sum(render_mem_occupancy) / len(render_mem_occupancy)}MB")
        elif camera_modality == 'ir':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras_ir[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)
                with torch.no_grad():
                    start_time = time.time()
                    rasterized_image, rasterized_depth, pixel_valid_point_count = self.rasteriser(
                        GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                            point_cloud=self.scene.point_cloud,
                            point_cloud_features=self.scene.point_cloud_features,
                            point_invalid_mask=self.scene.point_invalid_mask,
                            point_object_id=self.scene.point_object_id,
                            camera_info=camera_info_ir,
                            q_pointcloud_camera=q,
                            t_pointcloud_camera=t,
                            color_max_sh_band=3,
                        ),
                        current_train_stage='val_all',
                    )
                    end_time = time.time()
                    if i == 0:
                        pass
                    elif i > 0:
                        render_time.append(end_time - start_time)
                        render_mem_occupancy.append(get_gpu_mem())
                    rasterized_image_rgb = rasterized_image[:, :, :3]
                    rasterized_image_ms = rasterized_image[:, :, 3:4]
                    rasterized_image_ms = rasterized_image_ms.repeat(1, 1, 3)
                    rasterized_image_ir = rasterized_image[:, :, 4:]
                    rasterized_image_ir = rasterized_image_ir.repeat(1, 1, 3)

                    rasterized_image_rgb = torch.clamp(rasterized_image_rgb, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ms = torch.clamp(rasterized_image_ms, min=0, max=1).permute(2, 0, 1)
                    rasterized_image_ir = torch.clamp(rasterized_image_ir, min=0, max=1).permute(2, 0, 1)

                    image_rgb = self.ToPIL(rasterized_image_rgb)
                    image_ms = self.ToPIL(rasterized_image_ms)
                    image_ir = self.ToPIL(rasterized_image_ir)

                    image_rgb.save(os.path.join(save_dir_rgb, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ms.save(os.path.join(save_dir_ms, f'{TEST_ID[i].zfill(4)}.png'))
                    image_ir.save(os.path.join(save_dir_ir, f'{TEST_ID[i].zfill(4)}.png'))
            print(f"Average Render Time:{sum(render_time) / len(render_time)}s")
            print(f"Average CUDA Memory Occupancy: {sum(render_mem_occupancy) / len(render_mem_occupancy)}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default='bluechair')
    parser.add_argument("--render_view", type=str, default='ir', help="rendering view from which modality")
    parser.add_argument("--log_dir", type=str, default='./logs', help="parquet saving path")
    parser.add_argument("--output_prefix", type=str, default='./render')
    parser.add_argument("--poses_dir", type=str, default='./datasets', help="path that stored RGB poses")
    args = parser.parse_args()
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)
    print(f"Current rendering Dataset:{args.scene}, Current rendering View:{args.render_view}")

    # load parquet and poses
    parquet_path = os.path.join(args.log_dir, args.scene, 'best_scene.parquet')
    rgb_poses_path = os.path.join(args.poses_dir, args.scene, 'val.json')
    learned_poses_path = os.path.join(args.log_dir, args.scene)

    output_path = os.path.join(args.output_prefix, args.scene, args.render_view+'_view')
    output_gt_path = os.path.join(output_path, 'GT')
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_gt_path, exist_ok=True)

    # load optimized MS/IR pose(50k iterations)
    ms_pose_q = os.path.join(learned_poses_path, 'ms_iter_050000_r_.npy')
    ms_pose_t = os.path.join(learned_poses_path, 'ms_iter_050000_t_.npy')
    ir_pose_q = os.path.join(learned_poses_path, 'ir_iter_050000_r_.npy')
    ir_pose_t = os.path.join(learned_poses_path, 'ir_iter_050000_t_.npy')

    # poses for 30k iterations
    # ms_pose_q = os.path.join(learned_poses_path, 'ms_iter_030000_r_.npy')
    # ms_pose_t = os.path.join(learned_poses_path, 'ms_iter_030000_t_.npy')
    # ir_pose_q = os.path.join(learned_poses_path, 'ir_iter_030000_r_.npy')
    # ir_pose_t = os.path.join(learned_poses_path, 'ir_iter_030000_t_.npy')

    ms_pose_q_np = np.load(ms_pose_q)
    ms_pose_t_np = np.load(ms_pose_t)
    ir_pose_q_np = np.load(ir_pose_q)
    ir_pose_t_np = np.load(ir_pose_t)

    val_dataset = ImagePoseDataset(dataset_json_path=rgb_poses_path)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=4)

    cameras = torch.zeros((len(val_data_loader), 4, 4))
    cameras_ms = torch.zeros((len(val_data_loader), 4, 4))
    cameras_ir = torch.zeros((len(val_data_loader), 4, 4))
    camera_info = None

    for idx, val_data in enumerate(tqdm(val_data_loader)):
        image_gt, q_pointcloud_camera, t_pointcloud_camera, \
        image_gt_multispectral, q_pointcloud_camera_multispectral, t_pointcloud_camera_multispectral, \
        image_gt_infrared, q_pointcloud_camera_infrared, t_pointcloud_camera_infrared, \
        camera_info, camera_info_ms, camera_info_ir = val_data

        # image_gt, q, t, camera_info = val_data
        r_rgb = quaternion_to_rotation_matrix_torch(q_pointcloud_camera)
        cameras[idx, :3, :3] = r_rgb
        cameras[idx, :3, 3] = t_pointcloud_camera
        cameras[idx, 3, 3] = 1.0

        current_ms_pose_q_tensor = torch.tensor(ms_pose_q_np[camera_info.camera_id])
        current_r_ms = quaternion_to_rotation_matrix_torch(current_ms_pose_q_tensor)
        current_t_ms = torch.tensor(ms_pose_t_np[camera_info.camera_id])

        current_ir_pose_q_tensor = torch.tensor(ir_pose_q_np[camera_info.camera_id])
        current_r_ir = quaternion_to_rotation_matrix_torch(current_ir_pose_q_tensor)
        current_t_ir = torch.tensor(ir_pose_t_np[camera_info.camera_id])

        cameras_ms[idx, :3, :3] = current_r_ms
        cameras_ms[idx, :3, 3] = current_t_ms
        cameras_ms[idx, 3, 3] = 1.0

        cameras_ir[idx, :3, :3] = current_r_ir
        cameras_ir[idx, :3, 3] = current_t_ir
        cameras_ir[idx, 3, 3] = 1.0

        if args.render_view == 'rgb':
            img = torchvision.transforms.functional.to_pil_image(image_gt)
            img.save(os.path.join(output_gt_path, f'{TEST_ID[idx].zfill(4)}.png'))
        elif args.render_view == 'ms':
            img = torchvision.transforms.functional.to_pil_image(image_gt_multispectral)
            img.save(os.path.join(output_gt_path, f'{TEST_ID[idx].zfill(4)}.png'))
        elif args.render_view == 'ir':
            img = torchvision.transforms.functional.to_pil_image(image_gt_infrared)
            img.save(os.path.join(output_gt_path, f'{TEST_ID[idx].zfill(4)}.png'))

    config = GaussianPointRenderer.GaussianPointRendererConfig(
        parquet_path, cameras, cameras_ms, cameras_ir, camera_id=camera_info.camera_id
    )

    # override camera meta data as provided
    config.image_width = camera_info.camera_width
    config.image_height = camera_info.camera_height
    config.camera_intrinsics = camera_info.camera_intrinsics
    config.camera_intrinsics_ms = camera_info_ms.camera_intrinsics
    config.camera_intrinsics_ir = camera_info_ir.camera_intrinsics
    config.image_width_ms = camera_info_ms.camera_width
    config.image_height_ms = camera_info_ms.camera_height
    config.image_width_ir = camera_info_ir.camera_width
    config.image_height_ir = camera_info_ir.camera_height

    renderer = GaussianPointRenderer(config)
    renderer.run(output_path, camera_modality=args.render_view)

