# Cube Multi-View Reconstruction

A Python project that explores how two 2D camera views can be used to reconstruct a 3D cube. It combines custom perspective projection, synthetic training data, a PyTorch reconstruction model, viewable using a PySide6 interface.

## Features

- Cube translation, rotation, and scaling using vector and matrix operations.
- Perspective projection and movable cameras.
- Two camera views of the same cube.
- Synthetic training data with known 3D ground truth.
- CSV-based dataset storage.
- A PyTorch model for reconstruction from paired 2D observations.
- An integrated PySide6 interface with Matplotlib visualizations.

## Flow

1. **Cube Setup:** define the cube's eight vertices and apply 3D transformations.
2. **Mapping:** project the vertices into two camera views.
3. **Building the dataset:** pair the projected observations with the corresponding 3D coordinates.
4. **Train the model:** compare predicted coordinates with ground truth, calculate gradients through backpropagation, and update the model parameters.
5. **Inspect predictions:** display the camera views and reconstructed cube in the interface.

This is a supervised coordinate-reconstruction experiment using synthetic observations. It assumes corresponding cube vertices are available in both views; it does not detect a cube from photographs or recover arbitrary scenes.

## Limitations

- The model is trained on a restricted synthetic cube dataset.
- Predictions often fail when camera positions or cube configurations fall outside the training distribution.
- Accurate predictions within the training region do not establish reliable reconstruction for arbitrary viewpoints or objects.
- The project uses known vertex correspondences rather than image-based feature detection.
