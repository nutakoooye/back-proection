# импорт фукций для взаимодействия с интерфесом, созданном в Qt Cretor
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
ui_file = "UI/detail.ui"
loader = QUiLoader()
# загрузка интерфейса из файла и запись его в переменную ui
ui = loader.load(ui_file)

# Селекторы ( получение элементов интерфейса и запись их в переменную)
button_start = ui.findChild(QPushButton, "Start")
button_uploadFile = ui.findChild(QPushButton, "upload")

QTypeWinDn = ui.findChild(QGroupBox, 'TypeWinDn')
QTypeWinDp = ui.findChild(QGroupBox, 'TypeWinDp')

pathUi = ui.findChild(QLabel, 'Path')

selectedTypeWinDnArr = QTypeWinDn.findChildren(QRadioButton)
selectedTypeWinDpArr = QTypeWinDp.findChildren(QRadioButton)

comboBox_Dp = ui.findChild(QComboBox, 'comboBox_Dp')
comboBox_Dn = ui.findChild(QComboBox, 'comboBox_Dn')

outPutLabel = ui.findChild(QTextBrowser, 'Output')

# установка картинки весовой функции по умолчанию,
# чтобы при запуске интерфейса не было пустого окна
@Slot()
def QPrint(value):
    text = outPutLabel.toPlainText()
    outPutLabel.setText(f"{text}\n{value}")

# функция для начало расчета
def button_start_clicked():
    # Default values
    TypeWinDn = 1
    TypeWinDp = 1

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

    # функция получает значения с интерфейса и вызывает функцию основного расчета с
    # заданными параметрами, преведенными к необходимому типу, так как значения с полей приходит
    # строкой (Qt поддерживает только запятую в полях с типом double, поэтому
    # меняем меняем её на точку )
    returnedValues = {
        'Kss': int(ui.findChild(QSpinBox, "Kss").text()),
        'dxsint': float(ui.findChild(QDoubleSpinBox, "dxsint").text().replace(',', '.')),
        'dysint': float(ui.findChild(QDoubleSpinBox, "dysint").text().replace(',', '.')),
        'StepBright': float(ui.findChild(QDoubleSpinBox, "StepBright").text().replace(',', '.')),
        'Nxsint': int(ui.findChild(QSpinBox, "Nxsint").text()),
        'Nysint': int(ui.findChild(QSpinBox, "Nysint").text()),
        'Tsint': float(ui.findChild(QDoubleSpinBox, "Tsint").text().replace(',', '.')),
        'tauRli': float(ui.findChild(QDoubleSpinBox, "tauRli").text().replace(',', '.')),
        'RegimRsa': 1,
        'TypeWinDp': TypeWinDp,
        'TypeWinDn': TypeWinDn,
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

# функция для загрузки файлов
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
        ModelDateLabel = ui.findChild(QTextBrowser, 'textBrowserModalDate')
        RowsAndCulCount = ui.findChild(QTextBrowser, 'RowsAndCulCount')

        # формирование текста, который будет выведен в окно. Этот текст показывает что лежит в файле
        formData = f"высота орбиты РСА = {format(ModelDateContent[0])}\n" \
                   f"длина волны = {format(ModelDateContent[1])}\n" \
                   f"ширина главного лепестка ДН по азимуту = {format(ModelDateContent[2])}\n" \
                   f"ширина главного лепестка ДН по углу места = {format(ModelDateContent[3])}\n" \
                   f"длительность импульса = {format(ModelDateContent[4])}\n" \
                   f"период повторения = {format(ModelDateContent[5])}\n" \
                   f"ширина спектра сигнала = {format(ModelDateContent[6])}\n" \
                   f"частота дискретизации = {format(ModelDateContent[7])}\n" \
                   f"широта центра участка синтезирования = {format(ModelDateContent[8])}\n" \
                   f"параметр согласованного фильтра bzc=pi*df/T0 = {format(ModelDateContent[10])}\n" \
                   f"время задержки записи по отношению к началу периода = {format(ModelDateContent[11])}\n" \
                   f"число отсчетов по быстрому времени = {format(ModelDateContent[12])}\n" \
                   f"число периодов повторения = {format(ModelDateContent[13])}\n" \
                   f"число приемных каналов = {format(ModelDateContent[14])}\n" \
                   f"расстояние между фазовыми центрами приемных каналов = {format(ModelDateContent[15])}\n" \
                   f"скорость РСА = {format(ModelDateContent[16])}\n" \
                   f"время синтезирования = {format(ModelDateContent[17])}\n" \
                   f"момент начала получения траекторного сигнала = {format(ModelDateContent[18])}\n" \
                   f"дискретность данных в консорт-файле по координатам = {format(ModelDateContent[19])}\n" \
                   f"дискретность данных в консорт-файле по скорости = {format(ModelDateContent[20])}\n" \
                   f"дискретность данных в консорт-файле по углам = {format(ModelDateContent[21])}"
        ModelDateLabel.setText(formData)
        RowsAndCulCount.setText(
            f"Размер входного массива - {int(ModelDateContent[12])} x {int(ModelDateContent[13])}\n " \
            f"type Int16\n" \
            f"Размер файла Yts1 - {(int(ModelDateContent[12]) * int(ModelDateContent[13]) * 2 * 2) / 1024 / 1024} Мб")
        fileNameLabel = ui.findChild(QLabel, 'File_name')
        fileNameLabel.setText(str(file_path).split("/")[-1])
        QPrint(f"Выбран файл: {file_path}")

# назначение обработчика на элемент
# Например, назначаем на кнопку button_start вызов функции button_start_clicked по клику
button_uploadFile.clicked.connect(open_file)
button_start.clicked.connect(button_start_clicked)

# отображение интерфейса в отдельным окном
ui.show()
app.exec()
