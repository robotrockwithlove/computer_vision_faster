"""
Создано Тимофеем Мареевым
"""

import os
import yaml
import threading
import copy
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET


import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFile, QIODevice, Qt, QXmlStreamReader
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QLabel, QFileDialog


from project_structure import ProjectStructure as PS
from using_visual_api import UsingVisualAPI as UVAPI
from UI.ui import Ui_MainWindow


class Ui(QtWidgets.QDialog, Ui_MainWindow):
    def __init__(self, MainWindow):
        super().__init__()
        self.MainWindow = MainWindow
        self.setupUi(self.MainWindow)

        self.init_app()
        self.subscribe()

    def subscribe(self):
        """подписка на события UI"""
        self.pushButton_create_project.clicked.connect(self.pushButton_create_project_click)
        self.pushButton_select_project.clicked.connect(self.pushButton_select_project_click)
        self.pushButton_select_task.clicked.connect(self.pushButton_select_task_click)
        self.pushButton_select_model.clicked.connect(self.pushButton_select_model_click)
        self.pushButton_stop.clicked.connect(self.pushButton_stop_click)
        self.pushButton_add_data_folder.clicked.connect(self.pushButton_add_data_folder_click)
        self.pushButton_select_data_folder.clicked.connect(self.pushButton_select_data_folder_click)
        self.pushButton_add_class.clicked.connect(self.pushButton_add_class_click)
        self.pushButton_del_class.clicked.connect(self.pushButton_del_class_click)
        self.label_annot_image.mousePressEvent = self.label_annot_image_press
        self.label_annot_image.mouseMoveEvent = self.label_annot_image_mouse_move


        self.tabWidget.currentChanged.connect(self.tabWidget_changed)
        self.listWidget_model.itemClicked.connect(self.listWidget_model_itemClicked)
        self.tableWidget_annotation_classes.clicked.connect(self.tableWidget_annotation_classes_clicked)
        self.listWidget_files_for_annot.itemClicked.connect(self.listWidget_files_for_annot_itemCliked)


        #таймер для обновления информации
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.showCounter)
        self.timer.start(40)

    def init_app(self):
        #пути
        self.path_root_app = os.getcwd()
        self.path = Path(self.path_root_app)

        self.path_projects = self.path.joinpath('projects')
        self.path_datasets = self.path.joinpath('datasets')
        self.path_models = self.path.joinpath('models')
        self.path_data = ''
        self.path_images = ''
        self.path_labels = ''

        #списки
        self.list_projects = self.get_project_list()
        self.list_models = self.get_models_list()
        self.list_task, self.dictionary_task = self.get_task_list()
        self.list_data_folders = self.get_data_list()

        self.list_annotation_classes = []
        self.list_work_images = []
        self.list_work_labels = []

        #переменные
        self.proj_struct_filepath = ''
        self.current_project = ''
        self.current_task = ''
        self.current_data = ''

        # вкладка аннотирования
        self.current_annot_class = 0
        self.current_image_ration = 1
        self.current_annot_image = None
        self.number_annot_point = 0
        self.first_point = []
        self.second_point = []


        #функции
        self.edit_project_list()

        #экземпляры
        self.ps = PS()
        self.uvapi = UVAPI()
        self.worker = None

        #настройка UI
        self.tabWidget.setCurrentIndex(0)

        while self.gridLayout_2.count():
            item = self.gridLayout_2.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()






    # события пользовательского интерфейса
    def pushButton_create_project_click(self):
        new_pr_str = self.lineEdit_name_new_project.text()
        if not new_pr_str:
            QtWidgets.QMessageBox.warning(None, 'Внимание', 'Введите название')
            return
        if new_pr_str.isspace() or \
                ' ' in new_pr_str:
            QtWidgets.QMessageBox.warning(None, 'Внимание', 'БезПробеловПожалуйста')
            return
        if new_pr_str[0].isdigit():
            QtWidgets.QMessageBox.warning(None, 'Внимание', 'Не начинайте с цифр')
            return
        if self.contains_cyrillic(new_pr_str):
            QtWidgets.QMessageBox.warning(None, 'Внимание', 'Пожалуйста латиницей')
            return

        if new_pr_str in self.list_projects:
            QtWidgets.QMessageBox.warning(None, 'Внимание', 'Такое название уже есть')
            return
        else:
            new_proj = self.path_projects.joinpath(new_pr_str)
            new_proj.mkdir()
            print('Создан проект с именем', new_pr_str)
            self.edit_project_list()
            self.ps.create_proj(new_proj, new_pr_str)
            item = self.listWidget_list_project.findItems(new_pr_str, QtCore.Qt.MatchFlag.MatchContains)
            self.listWidget_list_project.setCurrentItem(item[0])


    def pushButton_select_project_click(self):
        self.current_project = self.listWidget_list_project.currentRow()
        name_proj = self.listWidget_list_project.item(self.listWidget_list_project.currentRow()).text()
        self.proj_struct_filepath = Path(self.path_projects.joinpath(name_proj), (name_proj + ".xml"))
        self.update_treeWidget(1)

    def pushButton_select_task_click(self):
        self.current_task = self.listWidget_task.currentRow()
        task = self.listWidget_task.item(self.listWidget_task.currentRow()).text()
        task = self.dictionary_task.get(task)
        self.ps.add_task(self.proj_struct_filepath, task)
        self.update_treeWidget(2)

    def pushButton_select_model_click(self):
        pass

    def pushButton_select_data_folder_click(self):
        self.current_data = self.listWidget_data_catalog.currentRow()
        data = self.listWidget_data_catalog.item(self.listWidget_data_catalog.currentRow()).text()
        self.ps.add_data(self.proj_struct_filepath, data)
        self.update_treeWidget(4)

    def pushButton_add_class_click(self):
        num_rows = self.tableWidget_annotation_classes.rowCount()
        self.tableWidget_annotation_classes.insertRow(num_rows)

        item = QtWidgets.QTableWidgetItem(str(num_rows))
        self.tableWidget_annotation_classes.setItem(num_rows, 0, item)

    def pushButton_del_class_click(self):
        last_row = self.tableWidget_annotation_classes.rowCount() - 1
        self.tableWidget_annotation_classes.removeRow(last_row)

    def pushButton_stop_click(self):
        self.uvapi.stop_cycle()

    def pushButton_add_data_folder_click(self):
        directory = QFileDialog.getExistingDirectory(self, 'Выбрать папку', '/')

        if directory is not None and directory != '':
            path = Path(directory)
            path_directory = Path(self.path_datasets, path.name)
            include_images = Path(path_directory, 'images')
            include_labels = Path(path_directory, 'labels')
        else:
            return

        if path.name not in self.get_data_list():
            os.mkdir(path_directory)
            os.mkdir(include_images)
            os.mkdir(include_labels)

        if path.name in self.get_data_list():
            self.copy_files(path, include_images)

    def copy_files(self, from_folder, to_folder):
        folder_from = from_folder
        folder_to = to_folder
        list_extension = ['.mp4', '.mov', '.avi', '.mkv', '.wmv',
                          '.3gp', '.3g2', '.mpg', '.mpeg', '.m4v',
                          '.h264', '.flv', '.rm', '.swf', '.vob',
                          '.jpg', '.png', '.bmp', '.ai', '.psd',
                          '.ico', '.jpeg', '.ps', '.svg', '.tif', '.tiff']
        for f in os.listdir(folder_from):
            path = Path(f)
            extension = path.suffix
            if extension.lower() in list_extension:
                if os.path.isfile(os.path.join(folder_from, f)):
                    shutil.copy(os.path.join(folder_from, f), os.path.join(folder_to, f))
                if os.path.isdir(os.path.join(folder_from, f)):
                    os.system(f'rd /S /Q {folder_to}\\{f}')
                    shutil.copytree(os.path.join(folder_from, f), os.path.join(folder_to, f))



    def listWidget_model_itemClicked(self, item):
        path_model = Path(self.path_models, item.text())
        path_input = Path(self.path_root_app, 'short_street.avi')
        #path_input = Path(self.path_root_app, 'traffic-mini.mp4')
        #path_input = Path(self.path_root_app, 'photo_2023-04-06_13-57-42.jpg')
        #path_input = Path(self.path_root_app, 'bus.jpg')
        #path_input = Path(self.path_root_app, '000000000650.jpg')
        self.tabWidget.setCurrentIndex(6)

        threading.Thread(target=self.uvapi.fast_start, args=(path_model, str(path_input)), daemon=True).start()

    def tableWidget_annotation_classes_clicked(self, index):
        self.current_annot_class = index


    def tabWidget_changed(self, index):
        if index == 0:
            pass
        elif index == 1:
            pass
        elif index == 2:
            self.edit_models_list()
        elif index == 3:
            self.edit_data_list()
        elif index == 4:
            dir_data = self.list_data_folders[self.current_data]
            self.path_data = Path(self.path_datasets, dir_data)
            path_file_yaml = Path(self.path_data, dir_data+'.yaml')
            if os.path.isfile(path_file_yaml):
                self.list_annotation_classes = self.load_labels(path_file_yaml)
            else:
                annot = {'names': {0: 'first_class'}}
                with open(path_file_yaml, 'w', encoding='utf-8') as file:
                    documents = yaml.dump(annot, file)

            self.tableWidget_annotation_classes.setColumnCount(2)
            self.tableWidget_annotation_classes.setRowCount(1)

            self.tableWidget_annotation_classes.setColumnWidth(0, 30)

            #первый индекс это строка, второй это столбец
            for i in range(len(self.list_annotation_classes)):
                item = QtWidgets.QTableWidgetItem(str(i))
                self.tableWidget_annotation_classes.setItem(i, 0, item)
                item = QtWidgets.QTableWidgetItem(self.list_annotation_classes[i])
                self.tableWidget_annotation_classes.setItem(i, 1, item)
                num_rows = self.tableWidget_annotation_classes.rowCount()
                self.tableWidget_annotation_classes.insertRow(num_rows)

            #получим список файлов для аннотирования
            self.path_images = Path(self.path_data, 'images')
            self.path_labels = Path(self.path_data, 'labels')

            self.list_work_images = self.get_files_from_img(self.path_images)
            self.list_work_labels = self.get_files_from_labels(self.path_images)

            #поместить список файлов listWidget
            self.edit_work_images_list(self.path_images)

            #если список аннотаций пуст создать его
            if len(self.list_work_labels) == 0:
                for i in range(len(self.list_work_images)):
                    path_file = Path(self.path_labels, str(self.list_work_images[i])+'.txt')
                    with open(path_file, "w") as file:
                        file.write('')

    def listWidget_files_for_annot_itemCliked(self, item):

        path_curren_file = Path(self.path_images, item.text())

        img = cv2.imread(str(path_curren_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.current_annot_image = copy.deepcopy(img)
        self.repaint_annot_image(self.current_annot_image)

    def repaint_annot_image(self, image):
        if image is not None:
            # W 871 H 731
            # следует сжать изображение чтоб оно помещалось в рамку

            w_picture_zone_max = 871
            h_picture_zone_max = 731

            img = copy.deepcopy(image)
            height = img.shape[0]
            width = img.shape[1]

            new_width = width
            new_height = height
            self.current_image_ration = 1

            if height > width:
                if img.shape[0] > h_picture_zone_max:
                    self.current_image_ration = img.shape[0] / h_picture_zone_max
                    new_width = int(width / self.current_image_ration)
                    new_height = int(height / self.current_image_ration)
            else:
                if img.shape[1] > w_picture_zone_max:
                    self.current_image_ration = img.shape[1] / w_picture_zone_max
                    new_width = int(width / self.current_image_ration)
                    new_height = int(height / self.current_image_ration)

            img = cv2.resize(img, (new_width, new_height))

            # сюда можно добавить обработку рамок аннотаций.

            #img = self.uvapi.draw(img, )


            qimage = QtGui.QImage(img.data,
                                  img.shape[1], img.shape[0],
                                  img.strides[0],
                                  QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimage)

            self.label_annot_image.resize(img.shape[1], img.shape[0])
            self.label_annot_image.setPixmap(pixmap)

    def label_annot_image_press(self, event):
        x = event.pos().x()
        y = event.pos().y()

        if self.number_annot_point == 0:
            self.first_point = [x, y]
            self.lineEdit_x1.setText(str(x))
            self.lineEdit_y1.setText(str(y))
            self.number_annot_point = 1

        elif self.number_annot_point == 1:
            self.second_point = [x, y]
            self.lineEdit_x2.setText(str(x))
            self.lineEdit_y2.setText(str(y))
            #фиксируем
            self.fix_annotation()

            self.number_annot_point = 0

    def label_annot_image_mouse_move(self, event):
        x = event.pos().x()
        y = event.pos().y()
        print(x, ' ', y)



    def showCounter(self):
        if self.tabWidget.currentIndex() == 0:
            pass
        elif self.tabWidget.currentIndex() == 1:
            pass
        elif self.tabWidget.currentIndex() == 2:
            pass
        elif self.tabWidget.currentIndex() == 4:
            self.repaint_annot_image(self.current_annot_image)

        elif self.tabWidget.currentIndex() == 6:
            pass

            img = self.uvapi.image
            image = copy.deepcopy(img)

            if image is not None:
                image_height, image_width, _ = image.shape
                aspect_ratio = image_width / image_height

                if aspect_ratio > 1.0:
                    self.label_video_capture.resize(800, 600)
                else:
                    self.label_video_capture.resize(600, 800)

                frame_width = self.label_video_capture.size().width()
                frame_height = self.label_video_capture.size().height()
                frame_aspect_ratio = frame_width / frame_height

                if aspect_ratio > frame_aspect_ratio:
                    new_width = frame_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = frame_height
                    new_width = int(new_height * aspect_ratio)

                out_image = cv2.resize(image, (new_width, new_height))
                out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)

                out_img = copy.deepcopy(out_image)
                qimage = QtGui.QImage(out_img.data, out_img.shape[1], out_img.shape[0], out_img.strides[0], QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimage)
                self.label_video_capture.setPixmap(pixmap)


    """Обработка логики приложения"""

    def printtree(self, s):
        tree = ET.fromstring(s)
        self.treeWidget_project_stuct.setColumnCount(1)
        a = QTreeWidgetItem([tree.tag])
        self.treeWidget_project_stuct.addTopLevelItem(a)

        def displaytree(a, s):
            for child in s:
                branch = QTreeWidgetItem([child.tag])
                a.addChild(branch)
                displaytree(branch, child)
            if s.text is not None:
                content = s.text
                a.addChild(QTreeWidgetItem([content]))

        displaytree(a, tree)

    def expand_all_items(self, treeItem):
        item1 = treeItem.invisibleRootItem()
        def expand(item):
            for i in range(item.childCount()):
                child = item.child(i)
                child.setExpanded(True)
                if child.childCount() > 0:
                    expand(child)

        expand(item1)

    def update_treeWidget(self, next_tab):
        self.treeWidget_project_stuct.clear()
        with open(self.proj_struct_filepath, 'r') as file:
            self.printtree(file.read())
        self.expand_all_items(self.treeWidget_project_stuct)
        self.tabWidget.setCurrentIndex(next_tab)

    def contains_cyrillic(self, text):
        for char in text:
            if char.isalpha() and char.lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
                return True
        return False

    """ 1. вкладка Название """
    def get_project_list(self):
        files = [x.name for x in self.path_projects.glob('**/*') if x.is_dir()]
        return files

    def edit_project_list(self):
        self.list_projects = self.get_project_list()
        self.listWidget_list_project.clear()
        self.listWidget_list_project.addItems(self.list_projects)

    """ 2. вкладка Задание """
    def get_task_list(self):
        list_item = [self.listWidget_task.item(row).text() for row in range(self.listWidget_task.count())]
        list_task = ['detection', 'classification', 'segmentation']
        dictionary_task = dict(zip(list_item, list_task))

        return list_item, dictionary_task

    """ 3. вкладка Модель """
    def get_models_list(self):
        files = [x.name for x in self.path_models.glob('*.*') if x.is_file()]
        return files

    def edit_models_list(self):
        self.list_models = self.get_models_list()
        self.listWidget_model.clear()
        self.listWidget_model.addItems(self.list_models)

    """ 4. вкладка Данные """
    def get_data_list(self):
        folders = [x.name for x in self.path_datasets.glob('*') if x.is_dir()]
        return folders

    def edit_data_list(self):
        self.list_data_folders = self.get_data_list()
        self.listWidget_data_catalog.clear()
        self.listWidget_data_catalog.addItems(self.list_data_folders)

    """ 5. вкладка Разметка """
    def load_labels(self, labels_file):
        with open(labels_file) as f:
            labels = []
            try:
                my_dict = yaml.load(f, Loader=yaml.FullLoader)
                labels = my_dict['names']
                labels = list(labels.values())
            except yaml.YAMLError as e:
                print(e)
                self.raise_error('The labels file has incorrect format.')
        return labels

    def get_files_from_img(self, path):
        files = [x.name for x in path.glob('*.*') if x.is_file()]
        return files

    def get_files_from_labels(self, path):
        files = [x.name for x in path.glob('*.*') if x.is_file()]
        return files

    def edit_work_images_list(self, path):
        self.list_work_images = self.get_files_from_img(path)
        self.listWidget_files_for_annot.clear()
        self.listWidget_files_for_annot.addItems(self.list_work_images)

    def fix_annotation(self):
        #нужно получить номер класса и координаты
        #и поместить в listWidget
        print(self.tableWidget_annotation_classes.currentRow())
        x1 = self.first_point[0]
        y1 = self.first_point[1]
        x2 = self.second_point[0]
        y2 = self.second_point[1]

        res = self.conv_bbox_to_yolo_style(x1, y1, x2, y2,
                                           self.current_annot_image,
                                           self.current_image_ration)

        out_str = str(self.tableWidget_annotation_classes.currentRow()) + \
                  ' ' + str(round(res[0], 6)) + \
                  ' ' + str(round(res[1], 6)) + \
                  ' ' + str(round(res[2], 6)) + \
                  ' ' + str(round(res[3], 6))

        self.listWidget_res_annotation.addItem(out_str)




    def conv_bbox_to_yolo_style(self, x1, y1, x2, y2, img, ratio):

        image_width = img.shape[1]
        image_height = img.shape[0]

        x1 = float(x1 * ratio)
        y1 = float(y1 * ratio)
        x2 = float(x2 * ratio)
        y2 = float(y2 * ratio)

        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
        if x1 <= x2:
            xc = x1 + (box_width / 2)
            yc = y1 + (box_height / 2)
        else:
            xc = x2 + (box_width / 2)
            yc = y2 + (box_height / 2)

        x = xc / image_width
        y = yc / image_height
        w = box_width / image_width
        h = box_height / image_height

        return [x, y, w, h]

    """ 6. вкладка  """
    """ 7. вкладка  """
    """ 8. вкладка  """




class calc_process:
    def __init__(self, path_model, path_input):
        self.uvapi = UVAPI()
        self.uvapi.fast_start(path_model, path_input)

        #self.stream = open_images_capture(src, loop)
        #self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            self.frame = self.uvapi.image

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True