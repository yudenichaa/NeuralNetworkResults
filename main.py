import os
import sys
from PyQt5 import QtWidgets
from AerialRoadsWidget import AerialRoadsWidget

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    aerial_roads_widget = AerialRoadsWidget()
    aerial_roads_widget.show()
    sys.exit(app.exec())
