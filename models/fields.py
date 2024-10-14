import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

# authors: This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        # GPT: Define the dimensions of each layer in the network.
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        # GPT: Initialize the positional encoding function to None.
        self.embed_fn_fine = None

        if multires > 0:
            # GPT: If using positional encoding, get the embedding function and update the input dimension.
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        # GPT: Store the total number of layers and the skip connections.
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        # GPT: Initialize the layers of the network.
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                # GPT: If this layer has a skip connection, adjust the output dimension.
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            # GPT: Create a linear layer.
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                # GPT: Perform geometric initialization for better convergence.
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                # GPT: Apply weight normalization to the layer.
                lin = nn.utils.weight_norm(lin)

            # GPT: Add the layer to the module.
            setattr(self, "lin" + str(l), lin)

        # GPT: Define the activation function.
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        # GPT: Scale the inputs.
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            # GPT: Apply positional encoding if enabled.
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        # GPT: Pass the inputs through each layer.
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                # GPT: Apply skip connection by concatenating inputs.
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                # GPT: Apply activation function except for the last layer.
                x = self.activation(x)
        # GPT: Adjust the output scaling.
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        # GPT: Compute the signed distance function.
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        # GPT: Compute the SDF along with feature vectors.
        return self.forward(x)

    def gradient(self, x):
        # GPT: Compute the gradient of the SDF with respect to the input coordinates.
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # GPT: Use autograd to compute gradients.
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


# authors: This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        # GPT: Store the rendering mode and output squeezing flag.
        self.mode = mode
        self.squeeze_out = squeeze_out

        # GPT: Define the dimensions of each layer.
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        # GPT: Initialize the positional encoding for view directions.
        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            if self.mode == "no_view_dir":
                dims[0] += 2 * (input_ch - 3)
            if self.mode == "ps":
                dims[0] = input_ch

        self.num_layers = len(dims)

        # GPT: Initialize the layers of the network.
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                # GPT: Apply weight normalization to the layer.
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        # GPT: Define the activation function.
        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        # GPT: Apply positional encoding if available.
        if self.embedview_fn is not None:
            points = self.embedview_fn(points)
            normals = self.embedview_fn(normals)
            view_dirs = self.embedview_fn(view_dirs)

        # GPT: Initialize the rendering input based on the mode.
        rendering_input = None

        if self.mode == 'idr':
            # GPT: In IDR mode, concatenate points, view directions, normals, and features.
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            # GPT: Exclude view directions from the input.
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            # GPT: Exclude normals from the input.
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        elif self.mode == "ps":
            # GPT: In photometric stereo mode, use only points.
            rendering_input = points

        x = rendering_input

        # GPT: Pass the input through each layer.
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                # GPT: Apply activation function except for the last layer.
                x = self.relu(x)

        if self.squeeze_out:
            # GPT: Apply sigmoid activation to the output.
            x = torch.sigmoid(x)
        return x


# authors: This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        # GPT: Store network parameters.
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            # GPT: Get positional encoding for input points.
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            # GPT: Get positional encoding for view directions.
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        # GPT: Store skip connections and view direction usage flag.
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # GPT: Define the layers for point processing.
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)]
        )

        # GPT: Define the layers for view direction processing.
        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])        

        if use_viewdirs:
            # GPT: Define layers for alpha, feature, and RGB outputs.
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            # GPT: Define a single output layer.
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            # GPT: Apply positional encoding to input points.
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            # GPT: Apply positional encoding to view directions.
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        # GPT: Pass through point processing layers.
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # GPT: Apply skip connections.
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # GPT: Compute alpha (density) and feature vector.
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            # GPT: Concatenate feature vector with view directions.
            h = torch.cat([feature, input_views], -1)

            # GPT: Pass through view direction processing layers.
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            # GPT: Compute RGB color.
            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        # GPT: Initialize the log variance as a learnable parameter.
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        # GPT: Compute the variance and return it for each input point.
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
