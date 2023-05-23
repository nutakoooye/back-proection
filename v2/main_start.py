from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication,
                               QPushButton,
                               QCheckBox,
                               QGroupBox,
                               QDoubleSpinBox,
                               QSpinBox,
                               QLabel,
                               QRadioButton,
                               QFileDialog,
                               QMainWindow)
from PySide6.QtCore import Slot
from v2.calc_rli import calc_rli
from v2.getFilesPath import getFilesPath

app = QApplication([])
window = QMainWindow()
ui_file = "UI/form.ui"
loader = QUiLoader()
ui = loader.load(ui_file)

# Селекторы
button_start = ui.findChild(QPushButton, "Start")
button_uploadFile = ui.findChild(QPushButton, "upload")

QTypeWinDn = ui.findChild(QGroupBox, 'TypeWinDn')
QTypeWinDp = ui.findChild(QGroupBox, 'TypeWinDp')

pathUi = ui.findChild(QLabel, 'Path')


@Slot()
def button_start_clicked():
    mode3 = ui.findChild(QRadioButton, 'Detailmode').isChecked()
    selectedTypeWinDnArr = QTypeWinDn.findChildren(QRadioButton)
    selectedTypeWinDpArr = QTypeWinDp.findChildren(QRadioButton)

    # Default values
    TypeWinDn = 1
    TypeWinDp = 1
    RegimRsa = 2
    isGPU = ui.findChild(QCheckBox, 'GPU').isChecked()
    # Ищем режим
    if int(mode3):
        RegimRsa = 1

    # Ищем режим TypeWinDn
    for i in range(len(selectedTypeWinDnArr)):
        if selectedTypeWinDnArr[i].isChecked():
            TypeWinDn = int(selectedTypeWinDnArr[i].objectName()[1:])

    # Ищем режим TypeWinDp
    for i in range(len(selectedTypeWinDpArr)):
        if selectedTypeWinDpArr[i].isChecked():
            TypeWinDp = int(selectedTypeWinDnArr[i].objectName()[1:])

    # Ищем пути к файлам
    ConsortPath, ModelDatePath, Yts1Path, Yts2Path = getFilesPath(str(pathUi.text()))
    returnedValues = {
        'Kss': int(ui.findChild(QSpinBox, "Kss").text()),
        'dxsint': float(ui.findChild(QDoubleSpinBox, "dxsint").text().replace(',', '.')),
        'dysint': float(ui.findChild(QDoubleSpinBox, "dysint").text().replace(',', '.')),
        'StepBright': float(ui.findChild(QDoubleSpinBox, "StepBright").text().replace(',', '.')),
        'Nxsint': int(ui.findChild(QSpinBox, "Nxsint").text()),
        'Nysint': int(ui.findChild(QSpinBox, "Nysint").text()),
        'Tsint': float(ui.findChild(QDoubleSpinBox, "Tsint").text().replace(',', '.')),
        'tauRli': float(ui.findChild(QDoubleSpinBox, "tauRli").text().replace(',', '.')),
        'RegimRsa': RegimRsa,
        'TypeWinDp': TypeWinDp,
        'TypeWinDn': TypeWinDn,
        'isGPU': isGPU,
        'FlagViewSignal': ui.findChild(QCheckBox, 'FlagViewSignal').isChecked(),
        'FlagWriteRli': ui.findChild(QCheckBox, 'FlagWriteRli').isChecked(),
        # 't_r_w': float(ui.findChild(QDoubleSpinBox, "t_r_w").text().replace(',', '.')),
        'ConsortPath': ConsortPath,
        'ModelDatePath': ModelDatePath,
        'Yts1Path': Yts1Path,
        'Yts2Path': Yts2Path
    }
    calc_rli(returnedValues)


def open_file():
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(window, "Выберите файл")

    if file_path:
        pathUi.setText(file_path)
        fileNameLabel = ui.findChild(QLabel, 'File_name')
        fileNameLabel.setText(str(file_path).split("/")[-1])
        print(f"Выбран файл: {file_path}")


button_uploadFile.clicked.connect(open_file)

button_start.clicked.connect(button_start_clicked)

ui.show()
app.exec()
