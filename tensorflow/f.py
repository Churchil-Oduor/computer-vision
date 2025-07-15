import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

def on_click():
    print("Button clicked!")

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("PyQt5 Demo")
window.setGeometry(100, 100, 300, 200)

layout = QVBoxLayout()

button = QPushButton("Click Me")
button.clicked.connect(on_click)

layout.addWidget(button)
window.setLayout(layout)

window.show()
sys.exit(app.exec_())

