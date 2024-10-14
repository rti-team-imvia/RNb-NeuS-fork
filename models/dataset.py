import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# authors: This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    #GPT: Load intrinsic (K) and extrinsic (R, t) camera parameters from a projection matrix P.
    if P is None:
        #GPT: If projection matrix P is not provided, read it from the specified file.
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            #GPT: Skip the first line if there are 4 lines (assuming it's a header).
            lines = lines[1:]
        #GPT: Extract the first four elements of each line to form the projection matrix.
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        #GPT: Convert the list of lists into a NumPy array of type float32.
        P = np.asarray(lines).astype(np.float32).squeeze()

    #GPT: Decompose the projection matrix P into intrinsic matrix K, rotation matrix R, and translation vector t.
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    #GPT: Normalize the intrinsic matrix K so that K[2,2] is 1.
    K = K / K[2, 2]
    #GPT: Initialize a 4x4 identity matrix for intrinsics and set the top-left 3x3 block to K.
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    #GPT: Initialize a 4x4 identity matrix for the camera pose.
    pose = np.eye(4, dtype=np.float32)
    #GPT: Set the rotation part of the pose matrix by transposing R.
    pose[:3, :3] = R.transpose()
    #GPT: Set the translation part of the pose matrix, converting t from homogeneous to Euclidean coordinates.
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    #GPT: Return the intrinsic and pose matrices.
    return intrinsics, pose

def load_image(path):
    #GPT: Load an image from the specified path without changing its bit depth.
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    #GPT: Determine the bit depth based on the data type of the image.
    if image.dtype == "uint8":
        bit_depth = 8
    elif image.dtype == "uint16":
        bit_depth = 16
    #GPT: Convert the image from BGR to RGB format and normalize pixel values to [0, 1].
    return cv.cvtColor(image, cv.COLOR_BGR2RGB) / np.float32(2**bit_depth - 1)

def load_normal(path):
    #GPT: Load a normal map from the specified path.
    image = load_image(path)
    #GPT: Scale the image values from [0, 1] to [-1, 1] to represent normals.
    normal = image * 2.0 - 1.0  # Convert to range [-1, 1]
    #GPT: Flip the y-axis and z-axis to match the coordinate system conventions.
    normal[:, :, 1] = -normal[:, :, 1]  # y-axis is flipped
    normal[:, :, 2] = -normal[:, :, 2]  # z-axis is flipped
    #GPT: Return the processed normal map.
    return normal

def save_image(path, image, bit_depth=8):
    #GPT: Save an image to the specified path with the given bit depth.
    image_cp = np.copy(image)
    #GPT: Scale the image pixel values to the specified bit depth.
    image_cp = (image_cp * np.float64(2**bit_depth - 1))
    #GPT: Clip the pixel values to ensure they are within the valid range.
    image_cp = np.clip(image_cp, 0, 2**bit_depth - 1)
    #GPT: Convert the image to the appropriate data type based on the bit depth.
    if bit_depth == 8:
        image_cp = image_cp.astype(np.uint8)
    elif bit_depth == 16:
        image_cp = image_cp.astype(np.uint16)
    #GPT: Convert the image from RGB to BGR format for OpenCV compatibility.
    image_cp = cv.cvtColor(image_cp, cv.COLOR_RGB2BGR)
    #GPT: Save the image using OpenCV without compression.
    cv.imwrite(path, image_cp, [cv.IMWRITE_PNG_COMPRESSION, 0])

def save_normal(path, normal, bit_depth=8):
    #GPT: Save a normal map to the specified path with the given bit depth.
    normal_flipped = np.copy(normal)
    #GPT: Flip the y-axis and z-axis back to match the image coordinate system.
    normal_flipped[:, :, 1] = -normal_flipped[:, :, 1]  # y-axis is flipped
    normal_flipped[:, :, 2] = -normal_flipped[:, :, 2]  # z-axis is flipped
    #GPT: Scale the normal values from [-1, 1] to [0, 1] for image saving.
    image = (normal_flipped + 1) / 2
    #GPT: Use the save_image function to save the normal map.
    save_image(path, image, bit_depth=bit_depth)


class Dataset:
    def __init__(self, conf, no_albedo=False):
        #GPT: Initialize the Dataset object with configuration and albedo settings.
        super(Dataset, self).__init__()
        print('Load data: Begin')
        #GPT: Set the device to CUDA (GPU).
        self.device = torch.device('cuda')
        #GPT: Store the configuration.
        self.conf = conf

        #GPT: Get data directories and settings from the configuration.
        self.data_dir = conf.get_string('data_dir')
        self.normal_dir = conf.get_string('normal_dir', default='normal')
        self.albedo_dir = conf.get_string('albedo_dir', default='')
        self.no_albedo = no_albedo
        if self.albedo_dir == '':
            #GPT: If no albedo directory is specified, set no_albedo to True.
            self.no_albedo = True
        self.mask_dir = conf.get_string('mask_dir', default='mask')

        #GPT: Get camera file names from the configuration.
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        
        #GPT: Get camera and scaling settings.
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        #GPT: Load the camera parameters from the specified file.
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        #GPT: Load and process mask images.
        masks_lis = sorted(glob(os.path.join(self.data_dir, self.mask_dir, '*.png')))
        #GPT: Read each mask image and normalize pixel values to [0, 1].
        masks_np = np.stack([cv.imread(im_name, -1) for im_name in masks_lis]) / 255.0
        #GPT: Binarize the masks; set values > 0.5 to 1.0, else 0.0.
        masks_np = np.where(masks_np > 0.5, 1.0, 0.0)
        #GPT: Store the number of images and image dimensions.
        self.n_images, self.H, self.W = masks_np.shape

        #GPT: Load and process normal maps.
        normals_lis = sorted(glob(os.path.join(self.data_dir, self.normal_dir, '*.png')))
        #GPT: Read each normal map using the load_normal function.
        normals_np = np.stack([load_normal(im_name) for im_name in normals_lis])  # [n_images, H, W, 3]
        self.normals_lis = normals_lis

        if not self.no_albedo:
            #GPT: If albedo data is available, load and process albedo images.
            albedos_lis = sorted(glob(os.path.join(self.data_dir, self.albedo_dir, '*.png')))
            #GPT: Read each albedo image using the load_image function.
            albedos_np = np.stack([load_image(im_name) for im_name in albedos_lis])
            self.albedos_lis = albedos_lis

        #GPT: Generate initial light directions (for warm-up).
        light_directions_cam_warmup_np = self.gen_light_directions().transpose()  # [n_lights, 3]
        self.n_lights = light_directions_cam_warmup_np.shape[0]
        #GPT: Compute shaded images for the warm-up phase.
        shaded_images_warmup_np = np.maximum(
            np.sum(normals_np[:, np.newaxis, :, :, :] * light_directions_cam_warmup_np[np.newaxis, :, np.newaxis, np.newaxis, :], axis=-1),
            0
        )[:, :, :, :, np.newaxis]
        if not self.no_albedo:
            #GPT: Multiply albedo with shading for warm-up images.
            images_warmup_np = albedos_np[:, np.newaxis, :, :, :] * shaded_images_warmup_np
        else:
            #GPT: If no albedo, replicate the shading to create RGB images.
            images_warmup_np = np.tile(shaded_images_warmup_np, (1, 1, 1, 1, 3))

        #GPT: Generate light directions for each pixel (if normals are provided).
        light_directions_cam_np = self.gen_light_directions(normals_np)  # [n_images, n_lights, H, W, 3]
        # shaded_images_np = np.maximum(np.sum(normals_np[:, np.newaxis, :, :, :] * light_directions_cam_np, axis=-1), 0)[:, :, :, :, np.newaxis]
        #GPT: Compute shaded images using per-pixel light directions.
        shaded_images_np = np.maximum(
            np.sum(normals_np[:, np.newaxis, :, :, :] * light_directions_cam_np, axis=-1),
            0
        )[:, :, :, :, np.newaxis]

        if not self.no_albedo:
            #GPT: Multiply albedo with shading for images.
            images_np = albedos_np[:, np.newaxis, :, :, :] * shaded_images_np
        else:   
            #GPT: If no albedo, replicate the shading to create RGB images.
            images_np = np.tile(shaded_images_np, (1, 1, 1, 1, 3))

        # authors: world_mat is a projection matrix from world to image
        #GPT: Load world matrices for all images.
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        # authors: scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        #GPT: Load scale matrices for all images.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        #GPT: Initialize lists to store intrinsics and poses.
        self.intrinsics_all = []
        self.pose_all = []
        #GPT: Initialize arrays to store light directions in world coordinates.
        light_directions_warmup_np = np.zeros((self.n_images, self.n_lights, 3))
        light_directions_np = np.zeros((self.n_images, self.n_lights, self.H, self.W, 3))
        #GPT: Loop over each image to compute camera parameters and transform light directions.
        for idx, scale_mat, world_mat in zip(range(self.n_images), self.scale_mats_np, self.world_mats_np):
            #GPT: Compute the projection matrix P.
            P = world_mat @ scale_mat
            P = P[:3, :4]
            #GPT: Decompose P into intrinsics and pose.
            intrinsics, pose = load_K_Rt_from_P(None, P)
            #GPT: Append the intrinsics and pose to the lists.
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

            #GPT: Convert light directions to world coordinates for warm-up.
            for idx_light in range(self.n_lights):
                light_directions_warmup_np[idx, idx_light, :] = np.matmul(
                    pose[:3, :3], light_directions_cam_warmup_np[idx_light, :].T
                ).T

                #GPT: Transform per-pixel light directions to world coordinates.
                light_dir_res = light_directions_cam_np[idx, idx_light, :, :, :].reshape(-1, 3)
                light_dir_world = np.matmul(pose[:3, :3], light_dir_res.T).T
                light_directions_np[idx, idx_light, :, :, :] = light_dir_world.reshape(self.H, self.W, 3)
            
        #GPT: Convert NumPy arrays to PyTorch tensors.
        self.light_directions_warmup = torch.from_numpy(light_directions_warmup_np.astype(np.float32)).cpu()  # [n_images, n_lights, 3]
        self.images_warmup = torch.from_numpy(images_warmup_np.astype(np.float32)).cpu()  # [n_images, n_lights, H, W, 3]
        self.light_directions = torch.from_numpy(light_directions_np.astype(np.float32)).cpu()  # [n_images, n_lights, H, W, 3]
        self.images = torch.from_numpy(images_np.astype(np.float32)).cpu()  # [n_images, n_lights, H, W, 3]
        self.masks = torch.from_numpy(masks_np.astype(np.float32)).unsqueeze(3).cpu()
        #GPT: Free up memory by deleting unused variables.
        del normals_np
        if not self.no_albedo:
            del albedos_np
        del light_directions_cam_warmup_np
        del light_directions_warmup_np
        del images_warmup_np
        del light_directions_cam_np
        del light_directions_np
        del images_np
        del masks_np
        #GPT: Stack intrinsics and poses into tensors and move to the device.
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # authors: Object scale mat: region of interest to **extract mesh**
        #GPT: Load the object's scale matrix.
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        #GPT: Transform the bounding box to the object's coordinate system.
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        #GPT: Store the transformed bounding box limits.
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')
    
    def gen_light_directions(self, normal=None):
        #GPT: Generate light directions, optionally based on normals.
        tilt = np.radians([0, 120, 240])
        slant = np.radians([30, 30, 30]) if normal is None else np.radians([54.74, 54.74, 54.74])
        n_lights = tilt.shape[0]

        #GPT: Compute the base light directions in camera space.
        u = -np.array([
            np.sin(slant) * np.cos(tilt),
            np.sin(slant) * np.sin(tilt),
            np.cos(slant)
        ])  # [3, n_lights]

        if normal is not None:
            #GPT: If normals are provided, adjust light directions per pixel.
            n_images, n_rows, n_cols, _ = normal.shape  # [n_images, H, W, 3]
            #GPT: Compute the outer product of normals at each pixel.
            outer_prod = np.einsum('...j,...k->...jk', normal, normal)  # [n_images, H, W, 3, 3]
            #GPT: Perform Singular Value Decomposition on the outer products.
            U, _, _ = np.linalg.svd(outer_prod)

            #GPT: Determine if the determinant of U is negative to handle reflections.
            det_U = np.linalg.det(U)
            det_U_sign = np.where(det_U < 0, -1, 1)[..., np.newaxis, np.newaxis]

            #GPT: Adjust U to ensure proper rotation matrices.
            R = np.where(det_U_sign < 0, 
                         np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])), 
                         np.einsum('...ij,jk->...ik', U, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])))
            
            #GPT: Further adjust R where the diagonal element is negative.
            R_22 = (R[..., 2, 2] < 0)[..., np.newaxis, np.newaxis]
            R = np.where(R_22, np.einsum('...ij,jk->...ik', R, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])), R)

            #GPT: Compute the final light directions by applying rotations.
            light_directions_all = np.einsum('...lm,mn->...ln', R, u)  # [n_images, H, W, 3, n_lights]
            #GPT: Rearrange axes to match expected dimensions.
            light_directions = light_directions_all.transpose(0, 4, 1, 2, 3)
        else:
            #GPT: If normals are not provided, use the base light directions.
            light_directions = u

        #GPT: Return the computed light directions.
        return light_directions

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        #GPT: Generate rays from a specific camera view at the given resolution level.
        l = resolution_level
        #GPT: Create a grid of pixel coordinates.
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        #GPT: Create homogeneous pixel coordinates.
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        #GPT: Unproject pixels to camera coordinates.
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        #GPT: Normalize the ray directions.
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        #GPT: Rotate ray directions to world coordinates.
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        #GPT: Get the ray origins (camera position).
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        #GPT: Return transposed rays and pixel coordinates.
        return (
            rays_o.transpose(0, 1),
            rays_v.transpose(0, 1),
            pixels_x.transpose(0, 1),
            pixels_y.transpose(0, 1)
        )

    def ps_gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        #GPT: Generate random pixel indices within the image dimensions.
        pixels_x = torch.randint(low=int(0.00 * self.W), high=int(1.00 * self.W), size=[batch_size])
        pixels_y = torch.randint(low=int(0.00 * self.H), high=int(1.00 * self.H), size=[batch_size])
        #GPT: Get the color and mask at the selected pixels.
        color = self.images[img_idx[0]][img_idx[1]][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx[0]][(pixels_y, pixels_x)]      # batch_size, 1
        #GPT: Create homogeneous pixel coordinates.
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        #GPT: Unproject pixels to camera coordinates.
        p = torch.matmul(self.intrinsics_all_inv[img_idx[0], None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Normalize the ray directions.
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        #GPT: Rotate ray directions to world coordinates.
        rays_v = torch.matmul(self.pose_all[img_idx[0], None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Get the ray origins (camera position).
        rays_o = self.pose_all[img_idx[0], None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        #GPT: Return the concatenated rays, colors, and mask.
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def ps_gen_random_rays_at_view_on_all_lights(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        #GPT: Generate random pixel indices within the image dimensions.
        pixels_x = torch.randint(low=int(0.00 * self.W), high=int(1.00 * self.W), size=[batch_size], device='cpu')
        pixels_y = torch.randint(low=int(0.00 * self.H), high=int(1.00 * self.H), size=[batch_size], device='cpu')
        #GPT: Get the images under different lighting conditions for warm-up and regular images.
        images_warmup = self.images_warmup[img_idx, :, pixels_y, pixels_x, :]  # n_lights, batch_size, 3
        images = self.images[img_idx, :, pixels_y, pixels_x, :]  # n_lights, batch_size, 3

        #GPT: Get the mask values at the selected pixels.
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 1
        #GPT: Create homogeneous pixel coordinates.
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().cuda()  # batch_size, 3
        #GPT: Unproject pixels to camera coordinates.
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Normalize the ray directions.
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        #GPT: Rotate ray directions to world coordinates.
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Get the ray origins (camera position).
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3

        #GPT: Return the rays, images, and pixel coordinates.
        return torch.cat([rays_o.cpu(), rays_v.cpu(), mask[:, :1].cpu()], dim=-1).cuda(), images_warmup.cuda(), images.cuda(), pixels_x.cuda(), pixels_y.cuda()

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        #GPT: Generate random pixel indices within the image dimensions.
        pixels_x = torch.randint(low=int(0.00 * self.W), high=int(1.00 * self.W), size=[batch_size])
        pixels_y = torch.randint(low=int(0.00 * self.H), high=int(1.00 * self.H), size=[batch_size])
        #GPT: Get the color and mask at the selected pixels.
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 1
        #GPT: Create homogeneous pixel coordinates.
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        #GPT: Unproject pixels to camera coordinates.
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Normalize the ray directions.
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        #GPT: Rotate ray directions to world coordinates.
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        #GPT: Get the ray origins (camera position).
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        #GPT: Return the concatenated rays, colors, and mask.
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        #GPT: Generate rays by interpolating between two camera poses.
        l = resolution_level
        #GPT: Create a grid of pixel coordinates.
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        #GPT: Create homogeneous pixel coordinates.
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        #GPT: Unproject pixels to camera coordinates.
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        #GPT: Normalize the ray directions.
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        #GPT: Interpolate translation between two poses.
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        #GPT: Get the inverse of the poses.
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        #GPT: Extract rotations from the poses.
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        #GPT: Create a Slerp (Spherical Linear Interpolation) object.
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        #GPT: Interpolate rotation.
        rot = slerp(ratio)
        #GPT: Construct the interpolated pose.
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        #GPT: Convert rotation and translation to tensors.
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        #GPT: Rotate ray directions to world coordinates.
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        #GPT: Get the ray origins (interpolated camera position).
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        #GPT: Return transposed rays.
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        #GPT: Compute near and far bounds for rays intersecting a sphere.
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        #GPT: Compute the midpoint along the ray where it is closest to the sphere center.
        mid = 0.5 * (-b) / a
        #GPT: Set near and far distances relative to the midpoint.
        near = mid - 1.0
        far = mid + 1.0
        #GPT: Return the near and far distances.
        return near, far

    def image_at(self, idx, resolution_level):
        #GPT: Load and resize an image at the specified index and resolution level.
        img = cv.imread(self.images_lis[idx])
        return cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
    
    def normal_at(self, idx, resolution_level):
        #GPT: Get the normal map at the specified index and resolution.
        normals = load_normal(self.normals_lis[idx]).reshape([-1, 3])
        #GPT: Transform normals to world coordinates using the pose.
        pose = self.pose_all[idx].detach().cpu().numpy()
        normals_world = np.matmul(pose[:3, :3], normals.T).T.reshape([self.H, self.W, 3])
        #GPT: Resize the normal map based on the resolution level.
        return cv.resize(normals_world, (self.W // resolution_level, self.H // resolution_level))
    
    def image_at_ps(self, idv, idl, resolution_level):
        img_warmup = self.images_warmup[idv,idl,:,:,:3].cpu().detach().numpy()
        img = self.images[idv,idl,:,:,:3].cpu().detach().numpy()
        return cv.resize(img_warmup, (self.W // resolution_level, self.H // resolution_level)), cv.resize(img, (self.W // resolution_level, self.H // resolution_level))
