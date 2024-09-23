# Embedder (`embedder.py`)

The `embedder.py` file defines a positional encoding mechanism inspired by the NeRF (Neural Radiance Fields) model. This positional encoding is used to map low-dimensional inputs (like 3D points or view directions) into a higher-dimensional space using periodic functions (sine and cosine). The purpose of this encoding is to enable neural networks to better capture high-frequency details, which is crucial for accurately modeling complex functions like surfaces or volumetric scenes.

## Key Components:

### 1. `Embedder` Class:
The `Embedder` class is responsible for creating a positional encoding based on the provided frequencies and periodic functions (sin and cos). The encoding maps the input coordinates into a higher-dimensional space, which improves the neural network's ability to represent high-frequency details in the data.

#### Constructor Parameters (`__init__`):
- `kwargs`: A dictionary that contains important parameters like:
  - `input_dims`: Dimensionality of the input (e.g., 3 for 3D coordinates).
  - `max_freq_log2`: Maximum frequency (log2 scale) for the encoding.
  - `num_freqs`: Number of different frequency bands to use.
  - `include_input`: Whether to include the original input in the encoding.
  - `log_sampling`: If `True`, frequencies are sampled logarithmically (as in NeRF); otherwise, they are sampled linearly.
  - `periodic_fns`: The periodic functions to apply (e.g., sine and cosine).

#### `create_embedding_fn()`:
This method generates a list of embedding functions. It first checks whether the original input should be included and then iterates over the frequency bands (based on the provided parameters). For each frequency, it applies the periodic functions (sin, cos) and stores these as lambda functions in the `embed_fns` list. The dimensionality of the output is also tracked using `out_dim`.

#### Frequency Bands:
- **Logarithmic Sampling (`log_sampling=True`)**: Frequencies are exponentially spaced between \(2^0\) and \(2^{\text{max\_freq\_log2}}\).
- **Linear Sampling (`log_sampling=False`)**: Frequencies are linearly spaced between \(2^0\) and \(2^{\text{max\_freq\_log2}}\).

#### `embed()`:
This method takes an input tensor and applies all the embedding functions stored in `embed_fns`. The results are concatenated to form a higher-dimensional output. Each periodic function applied to the input across different frequency bands contributes to capturing different levels of detail.

### 2. `get_embedder()` Function:
This function is a utility that creates an `Embedder` object based on the `multires` parameter, which determines how many frequency bands to use. It returns two things:
- An embedding function `embed(x)` that can be used to apply the positional encoding to an input tensor `x`.
- The dimensionality of the encoded output (`embedder_obj.out_dim`), which is useful for defining the input size of subsequent neural networks.

#### Parameters:
- `multires`: Controls the number of frequency bands used for encoding.
- `input_dims`: Specifies the dimensionality of the input (e.g., 3D coordinates or view directions).

## Summary of Workflow:

### Purpose:
The main goal of the `embedder.py` file is to enable neural networks to better represent high-frequency details in low-dimensional inputs like 3D coordinates or view directions. This is achieved by mapping the inputs to a higher-dimensional space using periodic functions (sine and cosine) over different frequency bands.

### How It Works:
- The `Embedder` class creates a set of embedding functions that apply sine and cosine transformations at different frequencies to the input.
- The transformed input captures both low- and high-frequency information, which helps the network model more complex patterns in 3D space, such as sharp edges or intricate surfaces.
- The `get_embedder()` function returns both the embedding function and the output dimensionality of the encoding, which can be plugged into neural networks for tasks like 3D rendering or scene reconstruction.

### Use Case:
This positional encoding is critical in neural rendering frameworks like NeRF and IDR, where accurately modeling fine geometric and appearance details is essential for generating high-quality renderings from neural representations. By enhancing the input representation with frequency-based encoding, neural networks can achieve much better performance in tasks that involve modeling detailed 3D structures and appearance from sparse or ambiguous data.
