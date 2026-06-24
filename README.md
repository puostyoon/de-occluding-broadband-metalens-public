# De-occluding broadband metalens

A computational imaging system that designs optimized metasurfaces and trains neural networks for robust imaging and restoration under obstructions (such as dirt, fence, dust).

## Overview

This project implements a **split-spectrum metasurface-based imaging pipeline** with **two-stage optimization**:

**Stage 1: Metasurface Design (Optical Part)**
- Learns a metasurface design optimized for de-occluding broadband imaging
- Uses wave optics based simulation (PADO library)
- Applies split-spectrum optimization with RGB bandpass filters
- **Insight**: Metasurface is designed to create favorable optical effects that make obstructions easier to remove

**Stage 2: Neural Network Restoration (Image Processing)**
- Trains a restoration network to further enhance optically degraded images

---

### Dataset Structure

The prepared dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1siNWBBHfKFFi4170p5rpSbTFk9WyrfSp?usp=sharing).

Prepare your dataset with the following folder structure for the neural network training:

```
dataset/
|-- de_occluding_broadband_metalens_training_data/
|   |-- Meta_camera/      (degraded images from metasurface)
|   |   |-- img_0000.png
|   |   |-- img_0001.png
|   |   `-- ...
|   `-- GT_camera/        (clean ground truth images)
|       |-- img_0000.png
|       |-- img_0001.png
|       `-- ...
`-- de_occluding_broadband_metalens_test_data/
    |-- Meta_camera/
    `-- GT_camera/
```

**Note**: Both `Meta_camera/` and `GT_camera/` must contain the same set of image files.

### Configuration Files

Configuration parameters are defined in `asset/config/param_*.py`.

---

## File Descriptions & Usage

### 1. `train_metasurface.sh`

**Purpose**: Optimize a metasurface for de-occluding broadband imaging.

**Function**: Bash wrapper script that calls `train_de_occluding_broadband_metalens.py` with preset hyperparameters.

**Usage**:
```bash
bash train_metasurface.sh
```
**Output**:
- Best DOE phase: `result_path/training_state_best.pt`
- Periodic checkpoints: `result_path/training_state_*.pt`
- TensorBoard logs: `result_path/runs/`

---

### 2. `train_neural_network.sh`

**Purpose**: Train a restoration neural network for enhanced image reconstruction.

**Function**: Bash wrapper for `train_neural_network_ddp.py` using `torchrun` for multi-GPU training.

**Note**: This stage is **independent of metasurface training**. The network learns to restore images degraded by the metasurface.

**Usage**:

```bash
bash train_neural_network.sh
```

**Output**:
- Best network weights: `result_path/training_state_minimum_eval_loss.pt`
- Periodic checkpoints: `result_path/training_state_*.pt`
- TensorBoard logs: `result_path/runs/`
- Training arguments: `result_path/args.json`

---

### 5. `example/inference_metasurface.ipynb`

**Purpose**: Visualize and analyze the learned metasurface design.

**Usage**:
1. Open `example/inference_metasurface.ipynb` in Jupyter
2. Set `ckpt_path` to trained metasurface directory
3. Run cells sequentially to visualize and analyze

---

### 6. `example/inference_neural_network.ipynb`

**Purpose**: Run and evaluate the trained restoration network.
**Usage**:
1. Open `example/inference_neural_network.ipynb` in Jupyter
2. Set `ckpt_path` to trained network checkpoint directory
3. Configure `positional_encoding` flag to match training setup
4. Select checkpoint filename (`saved_map_fname`)
5. Run cells to perform inference and save results
---

## Citation

If you use this project in your research, please cite:

```bibtex
Yoon, S. et al. De-occluding broadband metalens, version 1.0.0. Zenodo https://doi.org/10.5281/zenodo.20823451 (2026).
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://mit-license.org/) file for details.

---

## Acknowledgments

- [PADO library](https://github.com/shwbaek/pado) for differentiable wave optics simulation
- This project references code and logic for metalens training from [SeeThroughObstructions](https://github.com/princeton-computational-imaging/SeeThroughObstructions)
- This project references code for image reconstruction network from [ParamISP](https://github.com/woo525/ParamISP)

---

## Contact & Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [ysw1110@postech.ac.kr]

---

**Last Updated**: June 2026
