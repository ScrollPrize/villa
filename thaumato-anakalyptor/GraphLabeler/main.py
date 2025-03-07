import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from scroll_graph_util import ScrollGraph # Used to load .pkl for example in ome_zarr_view
from gui_main import PointCloudLabeler

def main():
    # Enable OpenGL rendering for pyqtgraph.
    pg.setConfigOptions(useOpenGL=False)
    app = QApplication(sys.argv)
    gui = PointCloudLabeler()  # Instantiate the main GUI window.
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
