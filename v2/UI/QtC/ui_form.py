# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.5.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QRadioButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1233, 672)
        Widget.setMinimumSize(QSize(1233, 0))
        self.horizontalLayoutWidget_4 = QWidget(Widget)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(30, 20, 711, 121))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.label_4 = QLabel(self.horizontalLayoutWidget_4)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_5.addWidget(self.label_4)

        self.spinBox_4 = QSpinBox(self.horizontalLayoutWidget_4)
        self.spinBox_4.setObjectName(u"spinBox_4")

        self.horizontalLayout_5.addWidget(self.spinBox_4)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.label_5 = QLabel(self.horizontalLayoutWidget_4)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_6.addWidget(self.label_5)

        self.doubleSpinBox_5 = QDoubleSpinBox(self.horizontalLayoutWidget_4)
        self.doubleSpinBox_5.setObjectName(u"doubleSpinBox_5")

        self.horizontalLayout_6.addWidget(self.doubleSpinBox_5)


        self.verticalLayout_2.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_6 = QLabel(self.horizontalLayoutWidget_4)
        self.label_6.setObjectName(u"label_6")

        self.horizontalLayout_7.addWidget(self.label_6)

        self.doubleSpinBox_6 = QDoubleSpinBox(self.horizontalLayoutWidget_4)
        self.doubleSpinBox_6.setObjectName(u"doubleSpinBox_6")

        self.horizontalLayout_7.addWidget(self.doubleSpinBox_6)


        self.verticalLayout_2.addLayout(self.horizontalLayout_7)


        self.horizontalLayout_4.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.label_7 = QLabel(self.horizontalLayoutWidget_4)
        self.label_7.setObjectName(u"label_7")

        self.horizontalLayout_8.addWidget(self.label_7)

        self.spinBox_7 = QSpinBox(self.horizontalLayoutWidget_4)
        self.spinBox_7.setObjectName(u"spinBox_7")

        self.horizontalLayout_8.addWidget(self.spinBox_7)


        self.verticalLayout_3.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.label_8 = QLabel(self.horizontalLayoutWidget_4)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_9.addWidget(self.label_8)

        self.spinBox_8 = QSpinBox(self.horizontalLayoutWidget_4)
        self.spinBox_8.setObjectName(u"spinBox_8")

        self.horizontalLayout_9.addWidget(self.spinBox_8)


        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_9 = QLabel(self.horizontalLayoutWidget_4)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_10.addWidget(self.label_9)

        self.spinBox_9 = QSpinBox(self.horizontalLayoutWidget_4)
        self.spinBox_9.setObjectName(u"spinBox_9")

        self.horizontalLayout_10.addWidget(self.spinBox_9)


        self.verticalLayout_3.addLayout(self.horizontalLayout_10)


        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.groupBox = QGroupBox(Widget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(30, 150, 351, 381))
        self.verticalLayoutWidget = QWidget(self.groupBox)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 24, 261, 361))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.radioButton_2 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_2.setObjectName(u"radioButton_2")

        self.verticalLayout.addWidget(self.radioButton_2)

        self.radioButton = QRadioButton(self.verticalLayoutWidget)
        self.radioButton.setObjectName(u"radioButton")

        self.verticalLayout.addWidget(self.radioButton)

        self.radioButton_3 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_3.setObjectName(u"radioButton_3")

        self.verticalLayout.addWidget(self.radioButton_3)

        self.radioButton_4 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_4.setObjectName(u"radioButton_4")

        self.verticalLayout.addWidget(self.radioButton_4)

        self.radioButton_5 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_5.setObjectName(u"radioButton_5")

        self.verticalLayout.addWidget(self.radioButton_5)

        self.radioButton_6 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_6.setObjectName(u"radioButton_6")

        self.verticalLayout.addWidget(self.radioButton_6)

        self.radioButton_7 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_7.setObjectName(u"radioButton_7")

        self.verticalLayout.addWidget(self.radioButton_7)

        self.radioButton_8 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_8.setObjectName(u"radioButton_8")

        self.verticalLayout.addWidget(self.radioButton_8)

        self.radioButton_9 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_9.setObjectName(u"radioButton_9")

        self.verticalLayout.addWidget(self.radioButton_9)

        self.radioButton_10 = QRadioButton(self.verticalLayoutWidget)
        self.radioButton_10.setObjectName(u"radioButton_10")

        self.verticalLayout.addWidget(self.radioButton_10)

        self.groupBox_2 = QGroupBox(Widget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(390, 150, 351, 381))
        self.verticalLayoutWidget_5 = QWidget(self.groupBox_2)
        self.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(10, 24, 261, 361))
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.radioButton_21 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_21.setObjectName(u"radioButton_21")

        self.verticalLayout_5.addWidget(self.radioButton_21)

        self.radioButton_22 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_22.setObjectName(u"radioButton_22")

        self.verticalLayout_5.addWidget(self.radioButton_22)

        self.radioButton_23 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_23.setObjectName(u"radioButton_23")

        self.verticalLayout_5.addWidget(self.radioButton_23)

        self.radioButton_24 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_24.setObjectName(u"radioButton_24")

        self.verticalLayout_5.addWidget(self.radioButton_24)

        self.radioButton_25 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_25.setObjectName(u"radioButton_25")

        self.verticalLayout_5.addWidget(self.radioButton_25)

        self.radioButton_26 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_26.setObjectName(u"radioButton_26")

        self.verticalLayout_5.addWidget(self.radioButton_26)

        self.radioButton_27 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_27.setObjectName(u"radioButton_27")

        self.verticalLayout_5.addWidget(self.radioButton_27)

        self.radioButton_28 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_28.setObjectName(u"radioButton_28")

        self.verticalLayout_5.addWidget(self.radioButton_28)

        self.radioButton_29 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_29.setObjectName(u"radioButton_29")

        self.verticalLayout_5.addWidget(self.radioButton_29)

        self.radioButton_30 = QRadioButton(self.verticalLayoutWidget_5)
        self.radioButton_30.setObjectName(u"radioButton_30")

        self.verticalLayout_5.addWidget(self.radioButton_30)

        self.groupBox_3 = QGroupBox(Widget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(780, 150, 391, 101))
        self.horizontalLayoutWidget = QWidget(self.groupBox_3)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(0, 19, 391, 81))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.radioButton_33 = QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_33.setObjectName(u"radioButton_33")

        self.horizontalLayout.addWidget(self.radioButton_33)

        self.radioButton_32 = QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_32.setObjectName(u"radioButton_32")

        self.horizontalLayout.addWidget(self.radioButton_32)

        self.radioButton_31 = QRadioButton(self.horizontalLayoutWidget)
        self.radioButton_31.setObjectName(u"radioButton_31")

        self.horizontalLayout.addWidget(self.radioButton_31)

        self.checkBox = QCheckBox(Widget)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setGeometry(QRect(900, 270, 281, 26))
        self.verticalLayoutWidget_6 = QWidget(Widget)
        self.verticalLayoutWidget_6.setObjectName(u"verticalLayoutWidget_6")
        self.verticalLayoutWidget_6.setGeometry(QRect(780, 310, 391, 119))
        self.verticalLayout_6 = QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_10 = QLabel(self.verticalLayoutWidget_6)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout_11.addWidget(self.label_10)

        self.spinBox_10 = QSpinBox(self.verticalLayoutWidget_6)
        self.spinBox_10.setObjectName(u"spinBox_10")

        self.horizontalLayout_11.addWidget(self.spinBox_10)


        self.verticalLayout_6.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.label_11 = QLabel(self.verticalLayoutWidget_6)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout_12.addWidget(self.label_11)

        self.spinBox_11 = QSpinBox(self.verticalLayoutWidget_6)
        self.spinBox_11.setObjectName(u"spinBox_11")

        self.horizontalLayout_12.addWidget(self.spinBox_11)


        self.verticalLayout_6.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.label_12 = QLabel(self.verticalLayoutWidget_6)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout_13.addWidget(self.label_12)

        self.spinBox_12 = QSpinBox(self.verticalLayoutWidget_6)
        self.spinBox_12.setObjectName(u"spinBox_12")

        self.horizontalLayout_13.addWidget(self.spinBox_12)


        self.verticalLayout_6.addLayout(self.horizontalLayout_13)

        self.checkBox_2 = QCheckBox(Widget)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(790, 500, 181, 26))
        self.pushButton = QPushButton(Widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(1070, 490, 101, 41))
        self.pushButton_2 = QPushButton(Widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(1010, 110, 161, 29))

        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.label_4.setText(QCoreApplication.translate("Widget", u"\u041a\u043e\u044d\u0444\u0444\u0438\u0446\u0438\u0435\u043d\u0442 \u043f\u0435\u0440\u0435\u0434\u0438\u0441\u043a\u0440\u0435\u0442\u0438\u0437\u0430\u0446\u0438\u0438", None))
        self.label_5.setText(QCoreApplication.translate("Widget", u"\u0414\u0438\u0441\u043a\u0440\u0435\u0442\u043d\u043e\u0441\u0442\u044c \u043f\u043e \u0434\u043e\u043b\u0433\u043e\u0442\u0435, \u043c", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"\u0414\u0438\u0441\u043a\u0440\u0435\u0442\u043d\u043e\u0441\u0442\u044c \u043f\u043e \u0448\u0438\u0440\u043e\u0442\u0435, \u043c", None))
        self.label_7.setText(QCoreApplication.translate("Widget", u"\u0421\u0442\u0435\u043f\u0435\u043d\u044c \u044f\u0440\u043a\u043e\u0441\u0442\u043d\u043e\u0433\u043e \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u044f", None))
        self.label_8.setText(QCoreApplication.translate("Widget", u"\u0427\u0438\u0441\u043b\u043e \u0442\u043e\u0447\u0435\u043a \u043f\u043e \u0434\u043e\u043b\u0433\u043e\u0442\u0435", None))
        self.label_9.setText(QCoreApplication.translate("Widget", u"\u0427\u0438\u0441\u043b\u043e \u0442\u043e\u0447\u0435\u043a \u043f\u043e \u0448\u0438\u0440\u043e\u0442\u0435", None))
        self.groupBox.setTitle(QCoreApplication.translate("Widget", u"\u0412\u0435\u0441\u043e\u0432\u0430\u044f \u0444\u0443\u043d\u043a\u0446\u0438\u044f \u043f\u043e \u043d\u0430\u043a\u043b\u043e\u043d\u043d\u043e\u0439 \u0434\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438", None))
        self.radioButton_2.setText(QCoreApplication.translate("Widget", u"\u043d\u0435\u0442", None))
        self.radioButton.setText(QCoreApplication.translate("Widget", u"\u043a\u043e\u0441\u0438\u043d\u0443\u0441", None))
        self.radioButton_3.setText(QCoreApplication.translate("Widget", u"\u043a\u043e\u0441\u0438\u043d\u0443\u0441 \u043a\u0432\u0430\u0434\u0440\u0430\u0442", None))
        self.radioButton_4.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430", None))
        self.radioButton_5.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430 (\u0442\u0440\u0435\u0442\u044c\u044f \u0441\u0442\u0435\u043f\u0435\u043d\u044c)", None))
        self.radioButton_6.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430 (\u0447\u0435\u0442\u0432\u0435\u0440\u0442\u0430\u044f \u0441\u0442\u0435\u043f\u0435\u043d\u044c)", None))
        self.radioButton_7.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=2.7", None))
        self.radioButton_8.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=3.1", None))
        self.radioButton_9.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=3.5", None))
        self.radioButton_10.setText(QCoreApplication.translate("Widget", u"\u0411\u043b\u0435\u043a\u043c\u0430\u043d\u0430-\u0425\u0435\u0440\u0440\u0438\u0441\u0430", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Widget", u"\u0412\u0435\u0441\u043e\u0432\u0430\u044f \u0444\u0443\u043d\u043a\u0446\u0438\u044f \u043f\u043e \u043f\u043e\u043f\u0435\u0440\u0435\u0447\u043d\u043e\u0439 \u0434\u0430\u043b\u044c\u043d\u043e\u0441\u0442\u0438", None))
        self.radioButton_21.setText(QCoreApplication.translate("Widget", u"\u043d\u0435\u0442", None))
        self.radioButton_22.setText(QCoreApplication.translate("Widget", u"\u043a\u043e\u0441\u0438\u043d\u0443\u0441", None))
        self.radioButton_23.setText(QCoreApplication.translate("Widget", u"\u043a\u043e\u0441\u0438\u043d\u0443\u0441 \u043a\u0432\u0430\u0434\u0440\u0430\u0442", None))
        self.radioButton_24.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430", None))
        self.radioButton_25.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430 (\u0442\u0440\u0435\u0442\u044c\u044f \u0441\u0442\u0435\u043f\u0435\u043d\u044c)", None))
        self.radioButton_26.setText(QCoreApplication.translate("Widget", u"\u0425\u0435\u043c\u043c\u0438\u043d\u0433\u0430 (\u0447\u0435\u0442\u0432\u0435\u0440\u0442\u0430\u044f \u0441\u0442\u0435\u043f\u0435\u043d\u044c)", None))
        self.radioButton_27.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=2.7", None))
        self.radioButton_28.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=3.1", None))
        self.radioButton_29.setText(QCoreApplication.translate("Widget", u"\u041a\u0430\u0439\u0437\u0435\u0440\u0430-\u0411\u0435\u0441\u0441\u0435\u043b\u044f, alfa=3.5", None))
        self.radioButton_30.setText(QCoreApplication.translate("Widget", u"\u0411\u043b\u0435\u043a\u043c\u0430\u043d\u0430-\u0425\u0435\u0440\u0440\u0438\u0441\u0430", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Widget", u"\u0420\u0435\u0436\u0438\u043c \u0441\u044a\u0435\u043c\u043a\u0438", None))
        self.radioButton_33.setText(QCoreApplication.translate("Widget", u"\u041c\u0430\u0440\u0448\u0440\u0443\u0442\u043d\u044b\u0439", None))
        self.radioButton_32.setText(QCoreApplication.translate("Widget", u"\u041c\u0430\u0440\u0448\u0440\u0443\u0442\u043d\u044b\u0439\n"
" \u0441\u043a\u043e\u0448\u0435\u043d\u043d\u044b\u0439", None))
        self.radioButton_31.setText(QCoreApplication.translate("Widget", u"\u0414\u0435\u0442\u0430\u043b\u044c\u043d\u044b\u0439", None))
        self.checkBox.setText(QCoreApplication.translate("Widget", u"\u041e\u0434\u043d\u043e\u043f\u0440\u043e\u0445\u043e\u0434\u043d\u0430\u044f \u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0440\u043e\u043c\u0435\u0442\u0440\u0438\u044f", None))
        self.label_10.setText(QCoreApplication.translate("Widget", u"\u0412\u0440\u0435\u043c\u044f \u0441\u0438\u043d\u0442\u0435\u0437\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u044f, c", None))
        self.label_11.setText(QCoreApplication.translate("Widget", u"\u0417\u0430\u0434\u0435\u0440\u0436\u043a\u0430 \u043c\u043e\u043c\u0435\u043d\u0442\u0430 \u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u0438\u044f \u0420\u041b\u0418-1, \u0441", None))
        self.label_12.setText(QCoreApplication.translate("Widget", u"\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u0432\u0440\u0435\u043c\u0435\u043d\u0438 \u043c\u0435\u0436\u0434\u0443 \u0420\u041b\u0418-1,-2. \u0441", None))
        self.checkBox_2.setText(QCoreApplication.translate("Widget", u"\u041e\u0442\u043e\u0431\u0440\u0430\u0436\u0430\u0442\u044c \u0441\u0438\u0433\u043d\u0430\u043b\u044b", None))
        self.pushButton.setText(QCoreApplication.translate("Widget", u"\u0420\u0430\u0441\u0447\u0435\u0442 \u0420\u041b\u0418", None))
        self.pushButton_2.setText(QCoreApplication.translate("Widget", u"\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0444\u0430\u0439\u043b \u0422\u0421", None))
    # retranslateUi

