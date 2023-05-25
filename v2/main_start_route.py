from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QApplication,
                               QPushButton,
                               QCheckBox,
                               QGroupBox,
                               QDoubleSpinBox,
                               QSpinBox,
                               QLabel,
QTextBrowser,
                               QRadioButton,
                               QFileDialog,
                                QComboBox,
                               QMainWindow)
from PySide6.QtCore import Slot
from PySide6.QtGui import QPixmap
from v2.calc_rli import calc_rli
from v2.getFilesPath import getFilesPath


app = QApplication([])
window = QMainWindow()
ui_file = "UI/route.ui"
loader = QUiLoader()
ui = loader.load(ui_file)

# Селекторы
button_start = ui.findChild(QPushButton, "Start")
button_uploadFile = ui.findChild(QPushButton, "upload")

QTypeWinDn = ui.findChild(QGroupBox, 'TypeWinDn')
QTypeWinDp = ui.findChild(QGroupBox, 'TypeWinDp')

pathUi = ui.findChild(QLabel, 'Path')

selectedTypeWinDnArr = QTypeWinDn.findChildren(QRadioButton)
selectedTypeWinDpArr = QTypeWinDp.findChildren(QRadioButton)

comboBox_Dp = ui.findChild(QComboBox, 'comboBox_Dp')
comboBox_Dn = ui.findChild(QComboBox, 'comboBox_Dn')

label_image_Dn = ui.findChild(QLabel, 'IMAGE1')
label_image_Dp = ui.findChild(QLabel, 'IMAGE2')

outPutLabel = ui.findChild(QTextBrowser, 'Output')

new_image_path = "UI/media/1.png"
new_pixmap = QPixmap(new_image_path)
label_image_Dn.setPixmap(new_pixmap)
label_image_Dp.setPixmap(new_pixmap)

@Slot()
def QPrint(value):
    text = outPutLabel.toPlainText()
    outPutLabel.setText(f"{text}\n{value}")

def button_start_clicked():
    # Default values
    # Ищем пути к файлам
    ConsortPath, ModelDatePath, Yts1Path, Yts2Path = getFilesPath(str(pathUi.text()))
    print(comboBox_Dp.currentIndex())


    returnedValues = {
        'Kss': int(ui.findChild(QSpinBox, "Kss").text()),
        'dxsint': float(ui.findChild(QDoubleSpinBox, "dxsint").text().replace(',', '.')),
        'dysint': float(ui.findChild(QDoubleSpinBox, "dysint").text().replace(',', '.')),
        'StepBright': float(ui.findChild(QDoubleSpinBox, "StepBright").text().replace(',', '.')),
        'Nxsint': int(ui.findChild(QSpinBox, "Nxsint").text()),
        'Nysint': int(ui.findChild(QSpinBox, "Nysint").text()),
        'Tsint': float(ui.findChild(QDoubleSpinBox, "Tsint").text().replace(',', '.')),
        'tauRli': float(ui.findChild(QDoubleSpinBox, "tauRli").text().replace(',', '.')),
        'RegimRsa': 2,
        'TypeWinDp': comboBox_Dp.currentIndex(),
        'TypeWinDn': comboBox_Dn.currentIndex(),
        'isGPU': ui.findChild(QCheckBox, 'GPU').isChecked(),
        'FlagViewSignal': ui.findChild(QCheckBox, 'FlagViewSignal').isChecked(),
        'FlagWriteRli': ui.findChild(QCheckBox, 'FlagWriteRli').isChecked(),
        'ConsortPath': ConsortPath,
        'ModelDatePath': ModelDatePath,
        'Yts1Path': Yts1Path,
        'Yts2Path': Yts2Path
    }

    print('Запуск расчета...')
    calc_rli(returnedValues, QPrint)


QPrint('Hello world!')
def open_file():
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(window, "Выберите файл")

    if file_path:
        pathUi.setText(file_path)
        ConsortPath, ModelDatePath, Yts1Path, Yts2Path = getFilesPath(str(pathUi.text()))
        ModelDateContent = []
        with open(ModelDatePath, 'r') as f:
            while True:
                # считываем строку
                line = f.readline()
                if not line:
                    break
                ModelDateContent.append(float(line))

        RowsAndCulCount = ui.findChild(QTextBrowser,'RowsAndCulCount')
        RowsAndCulCount.setText(f"Размер массива - {int(ModelDateContent[12])} x {int(ModelDateContent[13])}\n " \
                                f"Тип: Int16\n" \
                                f"Размер файла Yts1 - {(int(ModelDateContent[12])*int(ModelDateContent[13])*2*2)/1024/1024} Мб")
        fileNameLabel = ui.findChild(QLabel, 'File_name')
        fileNameLabel.setText(str(file_path).split("/")[-1])
        QPrint(f"Выбран файл: {file_path}")

def changeWindowFuncDp(index):
    label_image = ui.findChild(QLabel, 'IMAGE2')
    new_image_path = f"UI/media/{index-1}.png"
    new_pixmap = QPixmap(new_image_path)
    label_image.setPixmap(new_pixmap)

def changeWindowFuncDn(index):
    label_image = ui.findChild(QLabel, 'IMAGE1')
    new_image_path = f"UI/media/{index-1}.png"
    new_pixmap = QPixmap(new_image_path)
    label_image.setPixmap(new_pixmap)



button_uploadFile.clicked.connect(open_file)

comboBox_Dp.currentIndexChanged.connect(changeWindowFuncDp)
comboBox_Dn.currentIndexChanged.connect(changeWindowFuncDn)

button_start.clicked.connect(button_start_clicked)

ui.show()
app.exec()
