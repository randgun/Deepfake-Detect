#!/usr/bin/python
#coding: utf-8

import os, sys
from UI_class import MyUI
from PyQt5 import QtWidgets

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyUI()
    mainWindow.show()
    sys.exit(app.exec_())