
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QPushButton,QCheckBox,QGroupBox, QDoubleSpinBox, QSpinBox,QLabel, QRadioButton, QFileDialog,QMainWindow
from PySide6.QtCore import Slot
from v2.main import main
from v2.aaaaaa import getFilesPath
app = QApplication([])
window = QMainWindow()
ui_file = "form.ui"
loader = QUiLoader()
ui = loader.load(ui_file)

returnedValues = {
    'Kss': 4,
    'dxsint': 1.0,
    'dysint': 1.0,
    'StepBright': 1.0,
    'Nxsint': 201,
    'Nysint': 201,
    'Tsint': 0.4,
    'RegimRsa': 1,
    'tauRli': 0.00,
    't_r_w': 0.1, # мб не то :)
}
File_path = ''


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

    #Default values
    TypeWinDn = 1
    TypeWinDp = 1
    RegimRsa = 2

    #Ищем режим
    if mode3:
        RegimRsa = 1

    # Ищем режим TypeWinDn
    for i in range(len(selectedTypeWinDnArr)):
        if selectedTypeWinDnArr[i].isChecked():
            TypeWinDn = int(selectedTypeWinDnArr[i].objectName()[1:])

    # Ищем режим TypeWinDp
    for i in range(len(selectedTypeWinDpArr)):
        if selectedTypeWinDpArr[i].isChecked():
            TypeWinDp = int(selectedTypeWinDnArr[i].objectName()[1:])


    isGPU = ui.findChild(QCheckBox, 'GPU').isChecked()

    ConsortPath, ModelDatePath, Yts1Path, Yts2Path = getFilesPath(str(pathUi.text()))
    print(ConsortPath)
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
        't_r_w': float(ui.findChild(QDoubleSpinBox, "t_r_w").text().replace(',', '.')),
        'ConsortPath': ConsortPath,
        'ModelDatePath': ModelDatePath,
        'Yts1Path': Yts1Path,
        'Yts2Path': Yts2Path
        # мб не то :)
    }
    print('---Полученные значения от клиента---')
    print(returnedValues)


    main(returnedValues)

def open_file():
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(window, "Выберите файл")


    if file_path:
        pathUi.setText(file_path)
        fileNameLabel = ui.findChild(QLabel, 'File_name')
        File_path = file_path
        fileNameLabel.setText(str(file_path).split("/")[-1])

        print(f"Выбран файл: {file_path}")



button_uploadFile.clicked.connect(open_file)
# window.setCentralWidget(button)
# window.show()
button_start.clicked.connect(button_start_clicked)

ui.show()
app.exec()
