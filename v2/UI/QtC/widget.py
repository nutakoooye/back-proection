
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QPushButton, QDoubleSpinBox, QSpinBox,QLabel, QRadioButton, QFileDialog,QMainWindow
from PySide6.QtCore import Slot
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


@Slot()
def button_start_clicked():
    mode3 = ui.findChild(QRadioButton, 'Detailmode').isChecked()
    RegimRsa = 2
    if mode3:
        RegimRsa = 1
    print(ui.findChild(QRadioButton, 'Stripmap').isChecked())
    returnedValues = {
        'Kss': ui.findChild(QSpinBox, "Kss").text(),
        'dxsint': ui.findChild(QDoubleSpinBox, "dxsint").text(),
        'dysint': ui.findChild(QDoubleSpinBox, "dysint").text(),
        'StepBright': ui.findChild(QDoubleSpinBox, "StepBright").text(),
        'Nxsint': ui.findChild(QSpinBox, "Nxsint").text(),
        'Nysint': ui.findChild(QSpinBox, "Nysint").text(),
        'Tsint': ui.findChild(QDoubleSpinBox, "Tsint").text(),
        'tauRli': ui.findChild(QDoubleSpinBox, "tauRli").text(),
        'RegimRsa': RegimRsa,
        't_r_w': ui.findChild(QDoubleSpinBox, "t_r_w").text(), # мб не то :)
    }
    print('---Полученные значения от клиента---')
    print(f"{returnedValues}")


def open_file():
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(window, "Выберите файл")



    if file_path:
        pathUi = ui.findChild(QLabel, 'Path')
        pathUi.setText(file_path)
        fileName = ui.findChild(QLabel, 'File_name')
        fileName.setText(f"{file_path}".split("/")[-1])
        File_path = f'{file_path}'
        print(f"Выбран файл: {file_path}")



button_uploadFile.clicked.connect(open_file)
# window.setCentralWidget(button)
# window.show()
button_start.clicked.connect(button_start_clicked)

ui.show()
app.exec()
