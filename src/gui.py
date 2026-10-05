import math
import sys

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from projection.geometry import createCube
from projection.transforms import translatePoint, scalePoint, rotatePoint
from projection.camera import Camera
from projection.render import drawWireframe
from projection.triangulation import triangulatePoint

class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Multiview — Projection and Reconstruction")
        self.resize(1200, 800)

        self.points, self.edges = createCube()
        # Fixed parallel cameras satisfy the current triangulation assumptions.
        self.cameraA = Camera(position=(-1, 0, 0), angles=(0, 0, 0))
        self.cameraB = Camera(position=(1, 0, 0), angles=(0, 0, 0))
        self.controls = {}

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        panel = QWidget()
        panel.setMaximumWidth(260)
        panelLayout = QVBoxLayout(panel)
        self.addControls(panelLayout, "Rotation (degrees)", "rotation",
                         -180, 180, 0, 5)
        self.addControls(panelLayout, "Translation", "translation",
                         -10, 10, 0, 0.1)
        self.addControls(panelLayout, "Scale", "scale", 0.1, 3, 1, 0.1)

        reset = QPushButton("Reset cube")
        reset.clicked.connect(self.resetCube)
        panelLayout.addWidget(reset)
        note = QLabel("Cameras are fixed at X = −1 and X = +1.\n"
                      "Both face positive Z.\n\n"
                      "Coordinates outside the 256 × 256 image are cropped. "
                      "Edges crossing behind the camera are skipped.")
        note.setWordWrap(True)
        panelLayout.addWidget(note)
        panelLayout.addStretch()
        layout.addWidget(panel)

        self.figure = Figure(figsize=(9, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.subplots(1, 2)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Vertex", "Original X", "Original Y", "Original Z",
            "Recovered X", "Recovered Y", "Recovered Z", "3D error",
        ])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)

        self.summary = QLabel()
        results = QWidget()
        resultsLayout = QVBoxLayout(results)
        resultsLayout.addWidget(self.summary)
        resultsLayout.addWidget(self.table)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(results)
        splitter.setSizes([470, 280])
        layout.addWidget(splitter, 1)
        self.updateScene()

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

    def updateScene(self, *_):
        scale = [control.value() for control in self.controls["scale"]]
        angles = [math.radians(control.value())
                  for control in self.controls["rotation"]]
        translation = [control.value()
                       for control in self.controls["translation"]]

        transformed = []
        for point in self.points:
            point = scalePoint(*point, *scale)
            point = rotatePoint(*point, *angles)
            point = translatePoint(*point, *translation)
            transformed.append(point)

        projectedA = [self.cameraA.project(point) for point in transformed]
        projectedB = [self.cameraB.project(point) for point in transformed]
        for ax, projected, title in zip(
            self.axes, (projectedA, projectedB), ("Camera A", "Camera B")
        ):
            ax.clear()
            drawWireframe(projected, self.edges, ax=ax, title=title)
        self.canvas.draw_idle()

        self.table.setRowCount(len(transformed))
        errors = []
        for row, (original, pointA, pointB) in enumerate(
            zip(transformed, projectedA, projectedB)
        ):
            recovered = triangulatePoint(
                pointA, pointB, self.cameraA, self.cameraB)
            values = [str(row)] + [f"{value:.5f}" for value in original]
            if recovered is None:
                values += ["—", "—", "—", "Unavailable"]
            else:
                error = float(np.linalg.norm(np.asarray(recovered) - original))
                errors.append(error)
                values += [f"{value:.5f}" for value in recovered]
                values.append(f"{error:.3e}")
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))

        text = f"Reconstructed {len(errors)} / {len(transformed)} vertices"
        if errors:
            text += f"   |   Maximum 3D error: {max(errors):.3e} world units"
        self.summary.setText(text)


def main():
    app = QApplication(sys.argv)
    window = Viewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
