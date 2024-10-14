import torch
import torch.nn as nn

# authors: Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        #GPT: Store keyword arguments for later use.
        self.kwargs = kwargs
        #GPT: Create the embedding function based on the provided arguments.
        self.create_embedding_fn()

    def create_embedding_fn(self):
        #GPT: Initialize a list to hold embedding functions.
        embed_fns = []
        #GPT: Get the input dimension from kwargs.
        d = self.kwargs['input_dims']
        #GPT: Initialize the output dimension counter.
        out_dim = 0
        #GPT: If including the input in the embedding...
        if self.kwargs['include_input']:
            #GPT: Add the identity function to the embedding functions.
            embed_fns.append(lambda x: x)
            #GPT: Increment the output dimension by the input dimension.
            out_dim += d

        #GPT: Get the maximum frequency exponent.
        max_freq = self.kwargs['max_freq_log2']
        #GPT: Get the number of frequency bands.
        N_freqs = self.kwargs['num_freqs']

        #GPT: Generate frequency bands.
        if self.kwargs['log_sampling']:
            #GPT: If using log sampling, frequencies are spaced logarithmically.
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            #GPT: Else, frequencies are spaced linearly.
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        #GPT: For each frequency...
        for freq in freq_bands:
            #GPT: For each periodic function (e.g., sin, cos)...
            for p_fn in self.kwargs['periodic_fns']:
                #GPT: Add a function that applies the periodic function at the given frequency.
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                #GPT: Increment the output dimension by the input dimension.
                out_dim += d

        #GPT: Store the list of embedding functions.
        self.embed_fns = embed_fns
        #GPT: Store the total output dimension of the embedding.
        self.out_dim = out_dim

    def embed(self, inputs):
        #GPT: Apply each embedding function to the inputs and concatenate the results.
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    #GPT: Define embedding configuration.
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    #GPT: Create an Embedder instance with the specified configuration.
    embedder_obj = Embedder(**embed_kwargs)
    #GPT: Define an embedding function that uses the embed method of the Embedder instance.
    def embed(x, eo=embedder_obj): return eo.embed(x)
    #GPT: Return the embedding function and the output dimension of the embedding.
    return embed, embedder_obj.out_dim
