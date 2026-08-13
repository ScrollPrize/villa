### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

import sys
from PyQt5.QtWidgets import QApplication

# Custom GUI Elements
from GUI.MainWindow import ThaumatoAnakalyptor


def main():
    app = QApplication(sys.argv)
    ex = ThaumatoAnakalyptor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
