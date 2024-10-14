import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset, save_image, save_normal
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer

#GPT: Define the Runner class, which encapsulates the training and validation processes.
class Runner:
    def __init__(self, conf_path, mode='train_rnb', case='CASE_NAME', is_continue=False, no_albedo=False):
        #GPT: Set the device to GPU if available.
        self.device = torch.device('cuda')

        # Configuration
        #GPT: Store the configuration file path.
        self.conf_path = conf_path
        f = open(self.conf_path)
        #GPT: Read the configuration file.
        conf_text = f.read()
        #GPT: Replace the placeholder 'CASE_NAME' with the actual case name.
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        #GPT: Parse the configuration string using ConfigFactory.
        self.conf = ConfigFactory.parse_string(conf_text)
        #GPT: Update the data directory in the configuration with the actual case name.
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        #GPT: Set the base experiment directory.
        self.base_exp_dir = self.conf['general.base_exp_dir']
        #GPT: Create the base experiment directory if it doesn't exist.
        os.makedirs(self.base_exp_dir, exist_ok=True)
        #GPT: Initialize the dataset with the given configuration and albedo setting.
        self.dataset = Dataset(self.conf['dataset'], no_albedo)
        #GPT: Initialize the iteration step counter.
        self.iter_step = 0

        # Training parameters
        #GPT: Set the end iteration from the configuration.
        self.end_iter = self.conf.get_int('train.end_iter')
        #GPT: Set the warm-up iteration count.
        self.warm_up_iter = self.conf.get_int('train.warm_up_iter')
        #GPT: Set the frequency of saving checkpoints.
        self.save_freq = self.conf.get_int('train.save_freq')
        #GPT: Set the frequency of reporting progress.
        self.report_freq = self.conf.get_int('train.report_freq')
        #GPT: Set the frequency of validation.
        self.val_freq = self.conf.get_int('train.val_freq')
        #GPT: Set the frequency of validating the mesh.
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        #GPT: Set the batch size for training.
        self.batch_size = self.conf.get_int('train.batch_size')
        #GPT: Set the resolution level for validation images.
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        #GPT: Set the initial learning rate.
        self.learning_rate = self.conf.get_float('train.learning_rate')
        #GPT: Set the alpha value for learning rate scheduling.
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        #GPT: Determine whether to use a white background.
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        #GPT: Set the end iteration for warm-up of learning rate.
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        #GPT: Set the end iteration for annealing.
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        #GPT: Set the weight for the eikonal loss (implicit geometric regularization).
        self.igr_weight = self.conf.get_float('train.igr_weight')
        #GPT: Set the weight for the mask loss.
        self.mask_weight = self.conf.get_float('train.mask_weight')
        #GPT: Store whether to continue from a previous checkpoint.
        self.is_continue = is_continue
        #GPT: Store the mode of operation.
        self.mode = mode
        #GPT: Store whether albedo is used or not.
        self.no_albedo = self.dataset.no_albedo
        #GPT: Initialize a list to keep track of models.
        self.model_list = []
        #GPT: Initialize the TensorBoard writer to None.
        self.writer = None

        # Networks
        #GPT: Initialize a list to collect parameters to be trained.
        params_to_train = []
        #GPT: Create an instance of the NeRF model for rendering outside the object.
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        #GPT: Create the SDF network for representing the signed distance function.
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        #GPT: Create the deviation network for estimating the variance in volume rendering.
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        #GPT: Create the rendering network for predicting colors.
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        #GPT: Store the output dimension of the color network.
        self.color_depth = self.conf["model.rendering_network"]["d_out"]
        
        #GPT: Add the parameters of the NeRF model to the training list.
        params_to_train += list(self.nerf_outside.parameters())
        #GPT: Add the parameters of the SDF network.
        params_to_train += list(self.sdf_network.parameters())
        #GPT: Add the parameters of the deviation network.
        params_to_train += list(self.deviation_network.parameters())
        #GPT: If albedo is used, add the parameters of the color network.
        if not self.no_albedo:
            params_to_train += list(self.color_network.parameters())

        #GPT: Create an optimizer for the collected parameters.
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        #GPT: Initialize the renderer with the networks and configuration.
        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
        
        #GPT: Set the color depth attribute in the renderer.
        self.renderer.color_depth = self.color_depth

        # Load checkpoint
        #GPT: Initialize the latest model name to None.
        latest_model_name = None
        if is_continue:
            #GPT: If continuing, list all files in the checkpoints directory.
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                #GPT: Check if the file is a .pth file and within the iteration limit.
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    #GPT: Add valid model names to the list.
                    model_list.append(model_name)
            #GPT: Sort the model list to find the latest checkpoint.
            model_list.sort()
            #GPT: Get the latest model name.
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            #GPT: Log the checkpoint that is being loaded.
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            #GPT: Load the checkpoint.
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            #GPT: Backup code files for debugging purposes.
            self.file_backup()

    #GPT: Define the training function for the reflectance neural rendering.
    def train_rnb(self):
        #GPT: Initialize the TensorBoard writer.
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        #GPT: Update the learning rate according to the schedule.
        self.update_learning_rate()
        #GPT: Calculate the remaining steps to train.
        res_step = self.end_iter - self.iter_step
        #GPT: Get a random permutation of image indices.
        image_perm = self.get_image_perm()
        #GPT: Set the random seed for reproducibility.
        torch.random.manual_seed(0)
        #GPT: Start the training loop.
        for iter_i in tqdm(range(res_step)):
            #GPT: Set the seed for each iteration.
            torch.random.manual_seed(iter_i)
            #GPT: Get the current batch number from the image permutation.
            cbn = image_perm[self.iter_step % len(image_perm)]
            #GPT: Generate random rays and corresponding data from the dataset.
            data, true_rgb_warmup, true_rgb, pixels_x, pixels_y = self.dataset.ps_gen_random_rays_at_view_on_all_lights(cbn, self.batch_size)

            #GPT: Split the data into ray origins, directions, and mask.
            rays_o, rays_d, mask = data[:, :3], data[:, 3: 6], data[:, 6: 7]
            #GPT: Compute near and far bounds for the rays.
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            #GPT: Initialize background RGB to None.
            background_rgb = None
            if self.use_white_bkgd:
                #GPT: If using white background, set background RGB to ones.
                background_rgb = torch.ones([1, 3])

            if self.mask_weight > 0.0:
                #GPT: If mask weight is positive, create a binary mask.
                mask = (mask > 0.5).float()
            else:
                #GPT: Otherwise, use a mask of ones.
                mask = torch.ones_like(mask)
            #GPT: Compute the sum of the mask values.
            mask_sum = mask.sum() + 1e-5

            if self.iter_step < self.warm_up_iter:
                #GPT: During warm-up iterations, use the warm-up RGB values.
                true_rgb = true_rgb_warmup

                #GPT: Get the light directions for warm-up from the dataset.
                lights_dir = self.dataset.light_directions_warmup[cbn, :, :].cuda()
                #GPT: Reshape the light directions appropriately.
                lights_dir = lights_dir.reshape(self.dataset.n_lights, 1, 1, 3)

                #GPT: Render the scene using the renderer's warm-up function.
                render_out = self.renderer.render_rnb_warmup(
                    rays_o, rays_d, near, far, lights_dir,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    no_albedo=self.no_albedo
                )
            else:
                #GPT: For regular iterations, move pixel coordinates to CPU (if necessary).
                pixels_y = pixels_y.cpu()
                pixels_x = pixels_x.cpu()

                #GPT: Get the light directions from the dataset for each pixel.
                lights_dir = self.dataset.light_directions[cbn, :, pixels_y, pixels_x, :].cuda()  # [n_lights, batch_size, 3]
                #GPT: Reshape the light directions for batch processing.
                lights_dir = lights_dir.reshape(self.dataset.n_lights, self.batch_size, 1, 3)

                #GPT: Render the scene using the renderer's regular function.
                render_out = self.renderer.render_rnb(
                    rays_o, rays_d, near, far, lights_dir,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    no_albedo=self.no_albedo
                )

            #GPT: Extract outputs from the renderer.
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']
            normal = render_out['gradients']

            # Loss
            #GPT: Compute the color error, weighted by the mask.
            color_error = ((color_fine - true_rgb) * mask[None, :, :]).reshape(-1, self.color_depth)
            # authors: Compute the PSNR (commented out).
            # psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 *  mask[None, :, :]).sum() / ( mask[None, :, :] * 3.0)).sqrt())
            #GPT: Compute the color loss using L1 loss, normalized by the mask sum and number of lights.
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / (mask_sum * self.dataset.n_lights)

            #GPT: Set the eikonal loss from the gradient error.
            eikonal_loss = gradient_error

            #GPT: Compute the mask loss using binary cross-entropy.
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

            #GPT: Compute the total loss as a weighted sum of the individual losses.
            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight  # Note: Original comment about potential scaling.

            #GPT: Zero the gradients in the optimizer.
            self.optimizer.zero_grad()
            #GPT: Perform backpropagation to compute gradients.
            loss.backward()
            #GPT: Update the model parameters.
            self.optimizer.step()

            #GPT: Increment the iteration step.
            self.iter_step += 1

            #GPT: Log the losses and statistics to TensorBoard.
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
            self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
            self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
            # authors: Compute the PSNR (commented out).
            # self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                #GPT: Print the training progress at the specified frequency.
                print(self.base_exp_dir)
                print('iter:{:8>d} \nloss = {} \ncolor_loss={} \neikonal_loss={} \nmask_loss={} \nlr={}\n'.format(
                    self.iter_step, loss,
                    color_fine_loss,
                    eikonal_loss * self.igr_weight,
                    mask_loss * self.mask_weight,
                    self.optimizer.param_groups[0]['lr'])
                )

            if self.iter_step % self.save_freq == 0:
                #GPT: Save the model checkpoint at the specified frequency.
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                #GPT: Validate the model by rendering an image.
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                #GPT: Validate the mesh by extracting and saving it.
                self.validate_mesh()

            #GPT: Update the learning rate according to the schedule.
            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                #GPT: Get a new permutation of images after each epoch.
                image_perm = self.get_image_perm()

    #GPT: Get a random permutation of image indices.
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    #GPT: Compute the cosine annealing ratio for scheduling.
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    #GPT: Update the learning rate based on the iteration step.
    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            #GPT: Linearly increase the learning rate during warm-up.
            learning_factor = self.iter_step / self.warm_up_end
        else:
            #GPT: Use cosine annealing after warm-up.
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        #GPT: Update the learning rate for each parameter group in the optimizer.
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    #GPT: Backup code files and configurations for debugging.
    def file_backup(self):
        #GPT: Get the list of directories to backup from the configuration.
        dir_lis = self.conf['general.recording']
        #GPT: Create the recording directory.
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            #GPT: Create a directory for each source directory.
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            #GPT: List all files in the source directory.
            files = os.listdir(dir_name)
            for f_name in files:
                #GPT: Copy Python source files.
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        #GPT: Copy the configuration file to the recording directory.
        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    #GPT: Load a checkpoint from the specified model file.
    def load_checkpoint(self, checkpoint_name):
        #GPT: Load the checkpoint data.
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        #GPT: Restore the iteration step.
        self.iter_step = checkpoint['iter_step']
        #GPT: Load the state dictionaries for each network.
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        #GPT: If in shading mode, skip loading the color network.
        if "shading" not in self.mode:
            self.color_network.load_state_dict(checkpoint['color_network_fine'])
        #GPT: Load the optimizer state.
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        #GPT: Log that the loading is complete.
        logging.info('End')

    #GPT: Save the current model and optimizer state to a checkpoint.
    def save_checkpoint(self):
        #GPT: Create a dictionary containing the state dictionaries.
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }
        #GPT: Create the checkpoints directory if it doesn't exist.
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        #GPT: Save the checkpoint to a file named with the current iteration step.
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    #GPT: Validate the model by rendering an image.
    def validate_image(self, idv=-1, idl=-1, resolution_level=-1):
        if idv < 0:
            #GPT: Randomly select a camera index if not specified.
            idv = np.random.randint(self.dataset.n_images)
        if idl < 0:
            #GPT: Randomly select a light index if not specified.
            idl = np.random.randint(self.dataset.n_lights)

        #GPT: Print the validation iteration, camera, and light indices.
        print('Validate: iter: {}, camera: {}, light: {}'.format(self.iter_step, idv, idl))

        if resolution_level < 0:
            #GPT: Use the default resolution level if not specified.
            resolution_level = self.validate_resolution_level
        #GPT: Generate rays for the selected camera at the specified resolution.
        rays_o, rays_d, pixels_x, pixels_y = self.dataset.gen_rays_at(idv, resolution_level=resolution_level)
        #GPT: Get the image dimensions.
        H, W, _ = rays_o.shape

        #GPT: Round and convert pixel coordinates to integers.
        pixels_x = pixels_x.round().long()
        pixels_y = pixels_y.round().long()

        #GPT: Reshape and split the rays and pixel coordinates for batch processing.
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        pixels_x = pixels_x.reshape(-1, 1).split(self.batch_size)
        pixels_y = pixels_y.reshape(-1, 1).split(self.batch_size)

        #GPT: Initialize lists to collect rendered outputs.
        out_rgb_fine = []
        out_normal_fine = []

        #GPT: Iterate over batches of rays.
        for rays_o_batch, rays_d_batch, pixels_x_batch, pixels_y_batch in zip(rays_o, rays_d, pixels_x, pixels_y):
            #GPT: Compute near and far bounds for the batch.
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            #GPT: Set background RGB if using a white background.
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            if self.iter_step < self.warm_up_iter:
                #GPT: During warm-up, get the light directions for the selected camera and light.
                lights_dir = self.dataset.light_directions_warmup[idv, idl, :].cuda()
                #GPT: Reshape the light directions.
                lights_dir = lights_dir.reshape(1, 1, 1, 3)

                #GPT: Render the scene using the warm-up renderer.
                render_out = self.renderer.render_rnb_warmup(
                    rays_o_batch, rays_d_batch, near, far, lights_dir,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    no_albedo=self.no_albedo
                )
            else:
                #GPT: Move pixel coordinates to CPU (if necessary).
                pixels_x_batch = pixels_x_batch.cpu()
                pixels_y_batch = pixels_y_batch.cpu()

                #GPT: Get the light directions from the dataset for each pixel.
                lights_dir = self.dataset.light_directions[idv, idl, pixels_y_batch, pixels_x_batch, :].cuda().unsqueeze(0)
                #GPT: Render the scene using the regular renderer.
                render_out = self.renderer.render_rnb(
                    rays_o_batch, rays_d_batch, near, far, lights_dir,
                    background_rgb=background_rgb,
                    cos_anneal_ratio=self.get_cos_anneal_ratio(),
                    no_albedo=self.no_albedo
                )

            #GPT: Define a helper function to check if a key is in the render output and not None.
            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                #GPT: Collect the rendered colors.
                out_rgb_fine.append(render_out['color_fine'].squeeze(0).detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                #GPT: Compute normals by weighting gradients with ray weights.
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            #GPT: Delete the render output to free memory.
            del render_out

        #GPT: Initialize variables to store the final images.
        img_fine = None
        if len(out_rgb_fine) > 0:
            #GPT: Concatenate and reshape the rendered colors.
            img_fine = np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1])

        normal_img = None
        if len(out_normal_fine) > 0:
            #GPT: Concatenate and reshape the normals.
            normal_img = np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3, -1])

        #GPT: Create directories to save validation images and normals.
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        #GPT: Iterate over each output image in the batch.
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                if self.iter_step < self.warm_up_iter:
                    #GPT: Save the rendered image and ground truth during warm-up.
                    save_image(
                        os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idv, idl)),
                        np.concatenate([img_fine[..., i], self.dataset.image_at_ps(idv, idl, resolution_level=resolution_level)[0]])
                    )
                else:
                    #GPT: Save the rendered image and ground truth after warm-up.
                    save_image(
                        os.path.join(self.base_exp_dir, 'validations_fine', '{:0>8d}_{}_{}_{}.png'.format(self.iter_step, i, idv, idl)),
                        np.concatenate([img_fine[..., i], self.dataset.image_at_ps(idv, idl, resolution_level=resolution_level)[1]])
                    )
            if len(out_normal_fine) > 0:
                if self.iter_step < self.warm_up_iter:
                    #GPT: Save the rendered normals and ground truth during warm-up.
                    save_normal(
                        os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idv)),
                        np.concatenate([normal_img[..., i], self.dataset.normal_at(idv, resolution_level=resolution_level)])
                    )
                else:
                    #GPT: Save the rendered normals and ground truth after warm-up.
                    save_normal(
                        os.path.join(self.base_exp_dir, 'normals', '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idv)),
                        np.concatenate([normal_img[..., i], self.dataset.normal_at(idv, resolution_level=resolution_level)])
                    )

    #GPT: Render an image from a novel viewpoint by interpolating between two cameras.
    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        #GPT: Generate rays between the two camera indices at the given ratio.
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        #GPT: Get the image dimensions.
        H, W, _ = rays_o.shape
        #GPT: Reshape and split the rays for batch processing.
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        #GPT: Initialize a list to collect rendered colors.
        out_rgb_fine = []
        #GPT: Iterate over batches of rays.
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            #GPT: Compute near and far bounds.
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            #GPT: Set background RGB if using a white background.
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            #GPT: Render the scene using the renderer.
            render_out = self.renderer.render(
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                background_rgb=background_rgb
            )

            #GPT: Collect the rendered colors.
            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            #GPT: Delete the render output to free memory.
            del render_out

        #GPT: Concatenate and reshape the rendered colors to form the final image.
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 255).clip(0, 255).astype(np.uint8)
        return img_fine

    #GPT: Validate the mesh by extracting geometry and saving it.
    def validate_mesh(self, world_space=False, resolution=128, threshold=0.0):
        #GPT: Get the bounding box of the object.
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        #GPT: Extract the geometry (vertices and triangles) from the renderer.
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        #GPT: Create the meshes directory if it doesn't exist.
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            #GPT: Transform vertices to world space if required.
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        #GPT: Create a mesh from the vertices and triangles.
        mesh = trimesh.Trimesh(vertices, triangles)
        #GPT: Export the mesh to a PLY file named with the current iteration step.
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        #GPT: Log that the mesh validation is complete.
        logging.info('End')

    #GPT: Validate the mesh and extract texture (albedo).
    def validate_mesh_texture(self, resolution=128, threshold=0.0):
        #GPT: Get the bounding box of the object.
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        #GPT: Extract the geometry (vertices and triangles) from the renderer.
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        #GPT: Create the meshes directory if it doesn't exist.
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        #GPT: Transform vertices to world space.
        vertices_ws = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        # Get Albedo
        #GPT: Set the chunk size to process vertices in batches.
        cut_size = 100000
        #GPT: Split the vertices into chunks.
        vertices_tensor = torch.tensor(vertices).type(torch.float32).split(cut_size)
        #GPT: Initialize an array to store albedo values.
        albedo = np.empty(vertices.shape)
        #GPT: Iterate over each chunk of vertices.
        for k in range(len(vertices_tensor)):
            vt = vertices_tensor[k]
            #GPT: Get the SDF network output for the vertices.
            sdf_nn_output = self.sdf_network(vt)
            sdf = sdf_nn_output[:, :1]
            feature_vector = sdf_nn_output[:, 1:]

            #GPT: Compute gradients (normals) at the vertices.
            gradients = self.sdf_network.gradient(vt).squeeze()
            #GPT: Get the albedo from the color network and store it.
            albedo[k * cut_size:k * cut_size + vt.shape[0], :] = np.clip(
                1.00 * self.color_network(vt, gradients, gradients, feature_vector).cpu().detach().numpy()[:, [2, 1, 0]],
                0, 1
            )

        #GPT: Create a mesh with vertex colors (albedo).
        mesh = trimesh.Trimesh(vertices_ws, triangles, vertex_colors=albedo)
        #GPT: Export the textured mesh to a PLY file.
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        #GPT: Log that the textured mesh validation is complete.
        logging.info('End')

    #GPT: Interpolate between two camera views and create a video.
    def interpolate_view(self, img_idx_0, img_idx_1):
        #GPT: Initialize a list to store images.
        images = []
        n_frames = 60
        #GPT: Generate frames by interpolating between the two views.
        for i in range(n_frames):
            print(i)
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4
                )
            )
        #GPT: Append the reversed frames to create a looping video.
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        #GPT: Set up the video writer.
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(
            os.path.join(video_dir, '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
            fourcc, 30, (w, h)
        )

        #GPT: Write each image frame to the video file.
        for image in images:
            writer.write(image)

        #GPT: Release the video writer.
        writer.release()

#GPT: Entry point of the script.
if __name__ == '__main__':
    print('Hello Wooden')

    #GPT: Set the default tensor type to use GPU tensors.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    #GPT: Set up the logging format.
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    #GPT: Create an argument parser for command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/wmask_rnb.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--no_albedo', default=False, action="store_true")
    args = parser.parse_args()

    #GPT: Set the GPU device to use.
    torch.cuda.set_device(args.gpu)

    # BASE
    #GPT: Check the mode and execute the corresponding function.
    if args.mode == 'train_rnb':
        #GPT: Create a Runner instance with the given arguments.
        runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.no_albedo)
        #GPT: Start the training process.
        runner.train_rnb()
        #GPT: Validate the mesh after training.
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)

    elif args.mode == 'validate_mesh':
        #GPT: Create a Runner instance and validate the mesh.
        runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.no_albedo)
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh_texture':
        #GPT: Create a Runner instance and validate the textured mesh.
        runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.no_albedo)
        runner.validate_mesh_texture(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == "validate_image_ps":
        #GPT: Create a Runner instance and validate images.
        runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.no_albedo)
        runner.validate_image_ps()

# authors: 
# Train with reflectance
# python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME
# Train without reflectance
# python exp_runner.py --mode train_rnb --conf ./confs/CONF_NAME.conf --case CASE_NAME --no_albedo
# Extract surface
# python exp_runner.py --mode validate_mesh --conf ./confs/CONF_NAME.conf --case CASE_NAME --is_continue
