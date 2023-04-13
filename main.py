"""
Создано Тимофеем Мареевым
"""


import sys
from UI import ui
from procedure import Ui


if __name__ == '__main__':
    app = ui.QtWidgets.QApplication(sys.argv)
    MainWindow = ui.QtWidgets.QMainWindow()
    ui_inst = Ui(MainWindow)
    ui_inst.MainWindow.show()
    sys.exit(app.exec_())
