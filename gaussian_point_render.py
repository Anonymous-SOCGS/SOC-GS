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


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compute_pnsr_and_ssim(image_pred, image_gt):
    with torch.no_grad():
        psnr_score = 10 * \
                     torch.log10(1.0 / torch.mean((image_pred - image_gt) ** 2))
        image_pred_np = image_pred.permute(1, 2, 0).cpu().detach().numpy()
        image_gt_np = image_gt.permute(1, 2, 0).cpu().detach().numpy()
        ssim_score = ssim(image_pred_np, image_gt_np)
        # ssim_score = ssim(image_pred, image_gt, data_range=1.0, size_average=True)
        return psnr_score, ssim_score


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

        def set_portrait_mode(self):
            self.image_height = 976
            self.image_width = 544
            self.camera_intrinsics = torch.tensor(
                [[1163.486, 0.0, 272.0], [0.0, 1163.486, 488.0], [0.0, 0.0, 1.0]],
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
            camera_height_multispectral= self.config.image_height_ms,
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
        # the config does not matter here, only for training

        merged_point_cloud = torch.cat(
            [scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = torch.cat(
            [scene.point_cloud_features for scene in scene_list], dim=0)
        num_of_points_list = [scene.point_cloud.shape[0]
                              for scene in scene_list]
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
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.config.device)
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
        if camera_modality == 'RGB':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)

                with torch.no_grad():
                    image, _, _ = self.rasteriser(
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
                        current_train_stage='val_RGB',
                    )
                    image = torch.clamp(image, min=0, max=1).permute(2, 0, 1)
                    img2save = self.ToPIL(image)
                    img2save.save(output_prefix / f'frame_{i:03}.png')
        elif camera_modality == 'MS':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras_ms[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)
                with torch.no_grad():
                    # set MS rasterisation options
                    gaussian_point_cloud_rasterisation_input_ms = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ms,
                        q_pointcloud_camera=q,
                        t_pointcloud_camera=t,
                        color_max_sh_band=3,
                    )
                    rasterized_image_ms, rasterized_depth_ms, pixel_valid_point_count_ms = \
                        self.rasteriser(gaussian_point_cloud_rasterisation_input_ms,
                                           current_train_stage='val_MS')
                    image_pred_ms = rasterized_image_ms[:, :, -1].unsqueeze(dim=2)
                    image_pred_ms = torch.clamp(image_pred_ms, min=0, max=1)
                    image_pred_ms = image_pred_ms.permute(2, 0, 1)
                    image_pred_ms = image_pred_ms.repeat(3, 1, 1)
                    ms_image_pred = self.ToPIL(image_pred_ms)
                    ms_image_pred.save(output_prefix / f'ms_frame_{i:03}.png')
        elif camera_modality == 'IR':
            num_cameras = self.cameras.shape[0]
            for i in tqdm(range(num_cameras)):
                c = self.cameras_ir[i, :, :].unsqueeze(0)
                q, t = SE3_to_quaternion_and_translation_torch(c)
                with torch.no_grad():
                    # set IR rasterisation options
                    gaussian_point_cloud_rasterisation_input_ir = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_object_id=self.scene.point_object_id,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        camera_info=camera_info_ir,
                        q_pointcloud_camera=q,
                        t_pointcloud_camera=t,
                        color_max_sh_band=3
                    )
                    rasterized_image_ir, rasterized_depth_ir, pixel_valid_point_count_ir = \
                        self.rasteriser(gaussian_point_cloud_rasterisation_input_ir,
                                        current_train_stage='val_IR')
                    image_pred_ir = rasterized_image_ir[:, :, -1].unsqueeze(dim=2)
                    image_pred_ir = torch.clamp(image_pred_ir, min=0, max=1)
                    image_pred_ir = image_pred_ir.permute(2, 0, 1)
                    image_pred_ir = image_pred_ir.repeat(3, 1, 1)
                    # image_pred_ir = 255 * image_pred_ir
                    # image_pred_ir = torch.clamp(image_pred_ir, min=0, max=255)
                    ir_image_pred = self.ToPIL(image_pred_ir)
                    ir_image_pred.save(output_prefix / f'ir_frame_{i:03}.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--parquet_path", type=str, default='./logs/cvlab/best_scene.parquet')
    parser.add_argument("--parquet_path", type=str, default='./logs/cvlab/scene_50000.parquet')
    parser.add_argument("--poses", type=str, default='./datasets/cvlab/val.json', help="could be a .pt file that was saved as torch.save(), or a json dataset file used by Taichi-GS")
    parser.add_argument("--output_prefix", type=str, default='./result/cvlab/RGB')
    parser.add_argument("--gt_prefix", type=str, default="./result/cvlab/GT_RGB")
    parser.add_argument("--pose_path", type=str, default="./logs/cvlab/")
    parser.add_argument("--portrait_mode", action='store_true', default=False)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, device_memory_GB=4, kernel_profiler=True)

    output_prefix = Path(args.output_prefix)
    os.makedirs(output_prefix, exist_ok=True)
    # load optimized MS/IR pose
    ms_pose_q = os.path.join(args.pose_path, 'ms_iter_050000_r_.npy')
    ms_pose_t = os.path.join(args.pose_path, 'ms_iter_050000_t_.npy')
    ir_pose_q = os.path.join(args.pose_path, 'ir_iter_050000_r_.npy')
    ir_pose_t = os.path.join(args.pose_path, 'ir_iter_050000_t_.npy')
    # ms_pose_q = os.path.join(args.pose_path, 'ms_iter_150500_r_.npy')
    # ms_pose_t = os.path.join(args.pose_path, 'ms_iter_150500_t_.npy')
    # ir_pose_q = os.path.join(args.pose_path, 'ir_iter_150500_r_.npy')
    # ir_pose_t = os.path.join(args.pose_path, 'ir_iter_150500_t_.npy')
    ms_pose_q_np = np.load(ms_pose_q)
    ms_pose_t_np = np.load(ms_pose_t)
    ir_pose_q_np = np.load(ir_pose_q)
    ir_pose_t_np = np.load(ir_pose_t)

    if args.gt_prefix:
        gt_prefix = Path(args.gt_prefix)
        os.makedirs(gt_prefix, exist_ok=True)
    else:
        gt_prefix = None

    if args.poses.endswith(".pt"):
        config = GaussianPointRenderer.GaussianPointRendererConfig(
            args.parquet_path, torch.load(args.poses))
        if args.portrait_mode:
            config.set_portrait_mode()
    elif args.poses.endswith(".json"):
        val_dataset = ImagePoseDataset(dataset_json_path=args.poses)
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

            # dump autoscaled GT images at the resolution of training
            if gt_prefix is not None:
                img = torchvision.transforms.functional.to_pil_image(image_gt)
                img.save(gt_prefix / f'frame_{idx:03}.png')
        config = GaussianPointRenderer.GaussianPointRendererConfig(
            args.parquet_path, cameras, cameras_ms, cameras_ir, camera_id=camera_info.camera_id
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
    else:
        raise ValueError(f"Unrecognized poses file format: {args.poses}, Must be .pt or .json file")

    renderer = GaussianPointRenderer(config)
    renderer.run(output_prefix, camera_modality='RGB')



