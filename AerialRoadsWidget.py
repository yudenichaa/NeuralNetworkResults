import os
from PyQt5 import QtWidgets, QtCore, QtGui
from keras.engine.training import Model
from network import get_model, get_prediction
from tensorflow.python.framework.ops import Graph
from tensorflow.python.client.session import Session
from qimage2ndarray import array2qimage
from numpy import ndarray
from skimage.io import imread


class GetModelThread(QtCore.QThread):
    signal_model_loaded = QtCore.pyqtSignal(Model, Graph, Session)

    def __init__(self, model_path):
        QtCore.QThread.__init__(self)
        self.model_path = model_path

    def __del__(self):
        self.wait()

    def run(self):
        model, graph, session = get_model(self.model_path)
        self.signal_model_loaded.emit(model, graph, session)


class GetPredictionsThread(QtCore.QThread):
    signal_calculations_complete = QtCore.pyqtSignal(ndarray)

    def __init__(self, model, graph, session, image):
        QtCore.QThread.__init__(self)
        self.model = model
        self.graph = graph
        self.session = session
        self.image = image

    def __del__(self):
        self.wait()

    def run(self):
        prediction = get_prediction(
            self.model, self.graph,
            self.session, self.image)
        self.signal_calculations_complete.emit(prediction)


class AerialRoadsWidget(QtWidgets.QWidget):

    def _btn_choose_network_clicked(self):
        model_file_name = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Выберите модель',
            QtCore.QDir.currentPath(),
            'Keras model files (*.h5)')[0]

        if model_file_name:
            self.model_loading_thread = GetModelThread(model_file_name)
            self.model_loading_thread.signal_model_loaded.connect(self.slot_model_loaded)
            self.model_loading_thread.start()
            self.show_loading_indicator('Загрузка модели')

    def slot_model_loaded(self, model, graph, session):
        self.model = model
        self.graph = graph
        self.session = session
        self.hide_loading_indicator()

    def _btn_choose_folder_clicked(self):
        if not self.model:
            QtWidgets.QMessageBox.information(self, 'Информация', 'Модель не выбрана')
            return

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Выберите папку с изображениями',
            QtCore.QDir.currentPath())

        if directory:
            self.image_paths = []
            for file in sorted(os.listdir(directory)):
                if QtCore.QFileInfo(file).suffix() in ('png', 'jpg', 'jpeg', 'tiff'):
                    self.image_paths.append(os.path.join(directory, file))

            self.current_image_index = 0
            self.start_calculations()

    def start_calculations(self):
        image = imread(self.image_paths[self.current_image_index])
        self.calculations_thread = GetPredictionsThread(
            self.model, self.graph,
            self.session, image)
        self.calculations_thread.signal_calculations_complete.connect(self.slot_calculations_complete)
        self.calculations_thread.start()
        self.image = array2qimage(image)
        self.set_label_image(self.lbl_input_image_data, self.image)
        self.lbl_output_image_data.setMovie(self.loading_animation)
        self.loading_animation.start()

    def slot_calculations_complete(self, prediction):
        self.prediction = array2qimage(prediction)
        self.loading_animation.stop()
        self.set_label_image(self.lbl_output_image_data, self.prediction)

    def _btn_back_clicked(self):
        if len(self.image_paths) == 0:
            QtWidgets.QMessageBox.information(self, 'Информация', 'Выберите папку с изображениями')
            return

        if self.current_image_index == len(self.image_paths) - 1:
            self.current_image_index = 0
        else:
            self.current_image_index += 1

        self.start_calculations()

    def _btn_next_clicked(self):
        if len(self.image_paths) == 0:
            QtWidgets.QMessageBox.information(self, 'Информация', 'Выберите папку с изображениями')
            return

        if self.current_image_index == 0:
            self.current_image_index = len(self.image_paths) - 1
        else:
            self.current_image_index -= 1

        self.start_calculations()

    def _btn_restart_clicked(self):
        if len(self.image_paths) == 0:
            QtWidgets.QMessageBox.information(self, 'Информация', 'Выберите папку с изображениями')
            return

        self.current_image_index = 0
        self.start_calculations()

    def resizeEvent(self, event):
        self.setWindowState(QtCore.Qt.WindowMaximized)
        super().resizeEvent(event)

    def show_loading_indicator(self, message):
        self.layout_buttons.setContentsMargins(0, 0, 0, 0)
        self.lbl_loading.setText(message)
        self.lbl_loading_animation.setVisible(True)
        self.lbl_loading.setVisible(True)
        self.loading_animation.start()

    def hide_loading_indicator(self):
        self.layout_buttons.setContentsMargins(0, 0, 0, 25)
        self.lbl_loading_animation.setVisible(False)
        self.lbl_loading.setVisible(False)
        self.loading_animation.stop()

    @staticmethod
    def set_button_icon(button, image):
        icon = QtGui.QIcon(image)
        button.setIcon(icon)
        button.setIconSize(image.rect().size())

    def set_label_image(self, label, image):
        label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(
            self.image_label_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Технополис "ЭРА"')
        self.setWindowIcon(QtGui.QIcon('images/logo_era.ico'))
        self.setWindowState(QtCore.Qt.WindowFullScreen)

        self.image_label_size = QtCore.QSize(940, 880)
        self.model = None
        self.graph = None
        self.session = None
        self.calculations_thread = None
        self.model_loading_thread = None
        self.image_paths = []
        self.image = None
        self.prediction = None
        self.current_image_index = 0

        palette = self.palette()
        palette.setColor(QtGui.QPalette.Background, QtCore.Qt.white)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        btn_choose_folder = QtWidgets.QPushButton()
        btn_choose_folder.setFlat(True)
        btn_choose_network = QtWidgets.QPushButton()
        btn_choose_network.setFlat(True)
        btn_restart = QtWidgets.QPushButton()
        btn_restart.setFlat(True)
        btn_back = QtWidgets.QPushButton()
        btn_back.setFlat(True)
        btn_next = QtWidgets.QPushButton()
        btn_next.setFlat(True)

        folder_button_image = QtGui.QPixmap('images/folder_image48x48.png')
        network_button_image = QtGui.QPixmap('images/net48x48.png')
        restart_button_image = QtGui.QPixmap('images/restart48x48.png')
        back_button_image = QtGui.QPixmap('images/back48x48.png')
        next_button_image = QtGui.QPixmap('images/next48x48.png')

        AerialRoadsWidget.set_button_icon(btn_choose_folder, folder_button_image)
        AerialRoadsWidget.set_button_icon(btn_choose_network, network_button_image)
        AerialRoadsWidget.set_button_icon(btn_restart, restart_button_image)
        AerialRoadsWidget.set_button_icon(btn_back, back_button_image)
        AerialRoadsWidget.set_button_icon(btn_next, next_button_image)

        btn_choose_folder.clicked.connect(self._btn_choose_folder_clicked)
        btn_choose_network.clicked.connect(self._btn_choose_network_clicked)
        btn_restart.clicked.connect(self._btn_restart_clicked)
        btn_back.clicked.connect(self._btn_back_clicked)
        btn_next.clicked.connect(self._btn_next_clicked)

        label_font = QtGui.QFont('IMPACT', 13)
        lbl_input_image = QtWidgets.QLabel('Исходное изображение')
        lbl_input_image.setAlignment(QtCore.Qt.AlignCenter)
        lbl_input_image.setFont(label_font)
        lbl_output_image = QtWidgets.QLabel('Результат обработки')
        lbl_output_image.setAlignment(QtCore.Qt.AlignCenter)
        lbl_output_image.setFont(label_font)

        self.lbl_loading = QtWidgets.QLabel()
        self.lbl_loading.setVisible(False)
        self.lbl_loading.setContentsMargins(0, 35, 0, 0)
        self.lbl_loading.setFont(label_font)
        self.lbl_loading.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_loading_animation = QtWidgets.QLabel()
        self.lbl_loading_animation.setVisible(False)
        self.lbl_loading_animation.setAlignment(QtCore.Qt.AlignCenter)
        self.loading_animation = QtGui.QMovie('images/loading.gif')
        self.lbl_loading_animation.setMovie(self.loading_animation)

        self.lbl_input_image_data = QtWidgets.QLabel()
        self.lbl_output_image_data = QtWidgets.QLabel()
        self.lbl_input_image_data.setMaximumSize(self.image_label_size)
        self.lbl_output_image_data.setMaximumSize(self.image_label_size)
        self.lbl_input_image_data.setPixmap(QtGui.QPixmap('images/no_image.png').scaled(
            self.image_label_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))
        self.lbl_output_image_data.setPixmap(QtGui.QPixmap('images/no_image.png').scaled(
            self.image_label_size,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

        self.layout_buttons = QtWidgets.QHBoxLayout()
        self.layout_buttons.setContentsMargins(0, 0, 0, 25)
        self.layout_buttons.setAlignment(QtCore.Qt.AlignCenter)
        self.layout_buttons.addWidget(btn_choose_network)
        self.layout_buttons.addWidget(btn_choose_folder)
        self.layout_buttons.addWidget(btn_back)
        self.layout_buttons.addWidget(btn_next)
        self.layout_buttons.addWidget(btn_restart)

        layout_input_image = QtWidgets.QVBoxLayout()
        layout_input_image.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        layout_input_image.addWidget(lbl_input_image)
        layout_input_image.addWidget(self.lbl_input_image_data)

        layout_output_image = QtWidgets.QVBoxLayout()
        layout_output_image.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        layout_output_image.addWidget(lbl_output_image)
        layout_output_image.addWidget(self.lbl_output_image_data)

        layout_images = QtWidgets.QHBoxLayout()
        layout_images.addLayout(layout_input_image)
        layout_images.addLayout(layout_output_image)

        layout_main = QtWidgets.QVBoxLayout()
        layout_main.setAlignment(QtCore.Qt.AlignTop)
        layout_main.addLayout(self.layout_buttons)
        layout_main.addWidget(self.lbl_loading)
        layout_main.addWidget(self.lbl_loading_animation)
        layout_main.addLayout(layout_images)

        self.setLayout(layout_main)
