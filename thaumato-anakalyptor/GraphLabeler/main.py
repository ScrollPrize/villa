import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from gui_main import PointCloudLabeler

def main():
    # Enable OpenGL rendering for pyqtgraph.
    pg.setConfigOptions(useOpenGL=True)
    app = QApplication(sys.argv)
    gui = PointCloudLabeler()  # Instantiate the main GUI window.
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
