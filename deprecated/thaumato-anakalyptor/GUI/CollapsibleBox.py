### Julian Schilliger - ThaumatoAnakalyptor - Vesuvius 2024

from PyQt5.QtWidgets import QVBoxLayout, QGroupBox, QToolButton
from PyQt5.QtCore import Qt

class CollapsibleBox(QGroupBox):
    def __init__(self, title):
        super().__init__()
        self.initUI(title)

    def initUI(self, title):
        self.setFlat(True)  # Remove the box border

        # Toggle Button
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; text-align: left; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.clicked.connect(self.on_toggled)
        self.toggle_button.setCheckable(True)

        # Content Layout
        self.content_layout = QVBoxLayout()
        self.content_layout.setAlignment(Qt.AlignTop)
        
        # Main layout with button
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toggle_button)
        main_layout.addLayout(self.content_layout)

    def on_toggled(self):
        if self.toggle_button.isChecked():
            self.toggle_button.setArrowType(Qt.UpArrow)
        else:
            self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_content_visibility(self.toggle_button.isChecked())

    def toggle_content_visibility(self, visible):
        for i in range(self.content_layout.count()):
            widget = self.content_layout.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(visible)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)
        widget.setVisible(False)