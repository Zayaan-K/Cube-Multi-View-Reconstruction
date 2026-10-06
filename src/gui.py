"""Cube viewer with two camera views and learned 3D reconstruction.

Place in src/gui.py, with predict.py in src/model/.
Run from the project root: python src/gui.py
"""
import math
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QDoubleSpinBox, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

if __package__:
    from .projection.geometry import createCube
    from .projection.transforms import translatePoint, scalePoint, rotatePoint
    from .projection.camera import Camera
    from .projection.render import drawWireframe
    from .model.predict import ReconstructionPredictor
else:
    from projection.geometry import createCube
    from projection.transforms import translatePoint, scalePoint, rotatePoint
    from projection.camera import Camera
    from projection.render import drawWireframe
    from model.predict import ReconstructionPredictor


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Multiview — Neural Network Reconstruction")
        self.resize(1450, 850)
        self.points, self.edges = createCube()
        self.cameraA = Camera(position=(-1, 0, 0), angles=(0, 0, 0))
        self.cameraB = Camera(position=(1, 0, 0), angles=(0, 0, 0))
        self.checkpoint = Path(__file__).resolve().parent / "model" / "reconstruction.pt"
        self.predictor = None
        self.model_error = ""
        self.controls = {}
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        panel = QWidget()
        panel.setMaximumWidth(270)
        panel_layout = QVBoxLayout(panel)
        self.addControls(panel_layout, "Rotation (degrees)", "rotation", -180, 180, 0, 5)
        self.addControls(panel_layout, "Translation", "translation", -10, 10, 0, 0.1)
        self.addControls(panel_layout, "Scale", "scale", 0.1, 3, 1, 0.1)
        reset = QPushButton("Reset cube")
        reset.clicked.connect(self.resetCube)
        panel_layout.addWidget(reset)
        reload_button = QPushButton("Reload trained model")
        reload_button.clicked.connect(self.loadModel)
        panel_layout.addWidget(reload_button)
        note = QLabel(
            "Cameras are fixed at X = −1 and X = +1, facing positive Z.\n\n"
            "Training region: X/Y from −2 to +2; Z from 3 to 10. "
            "Predictions outside this region may be inaccurate.\n\n"
            "A vertex needs to be inside both 256 × 256 camera images."
        )
        note.setWordWrap(True)
        panel_layout.addWidget(note)
        panel_layout.addStretch()
        layout.addWidget(panel)

        self.figure = Figure(figsize=(12, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = [self.figure.add_subplot(1, 3, 1),
                     self.figure.add_subplot(1, 3, 2)]
        self.ax3d = self.figure.add_subplot(1, 3, 3, projection="3d")
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Vertex", "Actual X", "Actual Y", "Actual Z",
            "Predicted X", "Predicted Y", "Predicted Z", "3D error",
        ])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.addWidget(self.summary)
        results_layout.addWidget(self.table)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(results)
        splitter.setSizes([520, 280])
        layout.addWidget(splitter, 1)
        self.loadModel()

    def addControls(self, layout, title, key, minimum, maximum, default, step):
        group = QGroupBox(title)
        form = QFormLayout(group)
        values = []
        for axis in ("X", "Y", "Z"):
            spin = QDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(2)
            spin.setSingleStep(step)
            spin.setValue(default)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self.updateScene)
            form.addRow(axis, spin)
            values.append(spin)
        self.controls[key] = values
        layout.addWidget(group)

    def resetCube(self):
        for key, controls in self.controls.items():
            for control in controls:
                control.blockSignals(True)
                control.setValue(1 if key == "scale" else 0)
                control.blockSignals(False)
        self.updateScene()

    def loadModel(self, *_):
        self.predictor = None
        self.model_error = ""
        try:
            self.predictor = ReconstructionPredictor(self.checkpoint)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.model_error = f"Model unavailable: {error}"
        self.updateScene()

    @staticmethod
    def visible(point):
        return point is not None and all(np.isfinite(point)) and (
            0 <= point[0] < 256 and 0 <= point[1] < 256
        )

    def updateScene(self, *_):
        scale = [c.value() for c in self.controls["scale"]]
        angles = [math.radians(c.value()) for c in self.controls["rotation"]]
        translation = [c.value() for c in self.controls["translation"]]
        transformed = []
        for point in self.points:
            point = scalePoint(*point, *scale)
            point = rotatePoint(*point, *angles)
            transformed.append(translatePoint(*point, *translation))
        actual = np.asarray(transformed, dtype=float)
        projectedA = [self.cameraA.project(point) for point in actual]
        projectedB = [self.cameraB.project(point) for point in actual]
        for ax, projected, title in zip(self.axes, (projectedA, projectedB),
                                        ("Camera A", "Camera B")):
            ax.clear()
            drawWireframe(projected, self.edges, ax=ax, title=title)

        valid = [i for i, (a, b) in enumerate(zip(projectedA, projectedB))
                 if self.visible(a) and self.visible(b)]
        predicted = np.full(actual.shape, np.nan)
        prediction_error = self.model_error
        if self.predictor is not None and valid:
            inputs = [[*projectedA[i], *projectedB[i]] for i in valid]
            try:
                predicted[valid] = self.predictor.predict(inputs)
            except (RuntimeError, ValueError) as error:
                prediction_error = f"Prediction failed: {error}"
        available = np.isfinite(predicted).all(axis=1)
        distances = np.linalg.norm(predicted - actual, axis=1)
        self.table.setRowCount(len(actual))
        for row, point in enumerate(actual):
            values = [str(row)] + [f"{v:.5f}" for v in point]
            if available[row]:
                values += [f"{v:.5f}" for v in predicted[row]]
                values += [f"{distances[row]:.3e}"]
            else:
                values += ["—", "—", "—", "Unavailable"]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))

        ax = self.ax3d
        ax.clear()
        ax.scatter(*actual.T, color="tab:blue", label="Actual")
        if available.any():
            ax.scatter(*predicted[available].T, color="tab:orange", label="Predicted")
        for start, end in self.edges:
            ax.plot(*actual[[start, end]].T, color="tab:blue")
            if available[start] and available[end]:
                ax.plot(*predicted[[start, end]].T, color="tab:orange", linestyle="--")
        ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="3D reconstruction")
        ax.set_box_aspect((1, 1, 1))
        # Equal axis ranges preserve the cube's proportions.
        all_points = np.concatenate((actual, predicted[available]), axis=0)
        center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
        radius = max(float(np.ptp(all_points, axis=0).max()) * 0.6, 0.5)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
            setter(c - radius, c + radius)
        ax.legend()
        self.canvas.draw_idle()
        text = f"Predicted {int(available.sum())} / {len(actual)} vertices"
        if available.any():
            text += (f" | Mean 3D error: {distances[available].mean():.3e}"
                     f" | Maximum: {distances[available].max():.3e} world units")
        if prediction_error:
            text += f"\n{prediction_error}"
        outside = ((actual < (-2, -2, 3)) | (actual > (2, 2, 10))).any(axis=1)
        if outside.any():
            text += f"\n{int(outside.sum())} vertices are outside the training region."
        self.summary.setText(text)


def main():
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
