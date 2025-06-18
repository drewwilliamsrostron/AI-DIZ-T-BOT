"""Utility to create a PyQtGraph PlotWidget filling its parent."""

from PyQt5 import QtWidgets  # type: ignore
import pyqtgraph as pg


def create_plot(parent: QtWidgets.QWidget) -> pg.PlotWidget:
    """Return a PlotWidget embedded with zero margins."""
    plot = pg.PlotWidget(parent=parent)
    layout = QtWidgets.QVBoxLayout(parent)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(plot, stretch=1)
    parent.setLayout(layout)
    return plot
