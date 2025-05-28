# ARHMM Baby Project

An Autoregressive Hidden Markov Model (ARHMM) implementation for analyzing baby movement patterns using joint angle data.

## Overview

This project uses ARHMM to detect changepoints and analyze movement patterns in baby joint angle data. The model processes 8-dimensional joint angle data (shoulders, knees, hips, elbows) and identifies behavioral states and transitions.

## Requirements

### Hardware
- NVIDIA GPU with CUDA 12 support
- Sufficient RAM for data processing

### Software
- Python 3.10+
- CUDA 12.x installed system-wide
- pip package manager

## Installation

1. **Install CUDA 12** (if not already installed):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your operating system

2. **Clone/download this project**

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA installation**:
   ```bash
   nvcc --version  # Should show CUDA 12.x
   python -c "import jax; print('Devices:', jax.devices())"  # Should show GPU devices
   ```

## Data Format

### Input Data
Your input file must be a pickled Python dictionary where each entry maps a unique identifier to a corresponding time series of joint angle data:

- **Type**: dict[str, np.ndarray]
- **Key**: Baby label or name (e.g., "baby_01", "A", "subject_3")
- **Value**: A 2D NumPy array of joint angles
- **Shape**: (T, 8) — where T is the number of time points

### Labels
Create a corresponding label file with:
- **Alignment**: Each label index must correspond to the same individual in your angle data
- **Format**: Binary classification (0/1)
- **Note**: This is just for grapher.py to divide the results between the two groups

## Usage


### Basic Usage
Run the main analysis script:

```bash
python cp_finder_gpu.py
```

### Expected Output
The script will:
- Load and process your joint angle data
- Run ARHMM analysis to detect behavioral states
- Identify changepoints in movement patterns
- Generate analysis results and visualizations

## File Structure

- `cp_finder_gpu.py` - Main entry point for analysis
- `arhmm.py` - ARHMM implementation and analysis functions
- `requirements.txt` - Python package dependencies
- `baby_angles.pkl` - Example data file (if included)
- Additional supporting files for data processing and visualization

## Troubleshooting

### CUDA Issues
If you see warnings about CUDA not being found:
1. Verify CUDA 12 is installed: `nvcc --version`
2. Check JAX can see GPU: `python -c "import jax; print(jax.devices())"`
3. Reinstall JAX with CUDA support if needed

### Memory Issues
- Reduce data size if running out of GPU memory
- Consider processing data in smaller batches

### Data Format Issues
- Ensure your angle data has exactly 8 dimensions
- Verify labels and data have matching lengths
- Check for NaN or infinite values in your data

## Notes
- The model expects 8-dimensional joint angle input specifically
- GPU acceleration requires CUDA 12 compatibility
- Processing time depends on data size and GPU capabilities