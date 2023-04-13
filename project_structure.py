
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ProjectStructure(object):
    def __init__(self):
        pass

    def create_proj(self, path, name):
        project = ET.Element('project')

        name_project = ET.SubElement(project, 'name_project')
        name_project.text = name

        new_proj = path.joinpath(name+'.xml')

        data = self.prettify(project)
        with new_proj.open('w') as file:
            file.write(data)

    def add_task(self, path_file, value):
        tree = ET.parse(path_file, parser=ET.XMLParser(encoding='utf-8'))
        project = tree.getroot()

        task = project.find('task')
        if task is None:
            task = ET.SubElement(project, 'task')
            task.text = value
        else:
            task.text = value

        data = self.new_prettify(project)
        with open(path_file, 'w') as file:
            file.write(data)

    def add_model(self, path, value):
        pass

    def add_data(self, path_file, value):
        tree = ET.parse(path_file, parser=ET.XMLParser(encoding='utf-8'))
        project = tree.getroot()

        task = project.find('data')
        if task is None:
            task = ET.SubElement(project, 'data')
            task.text = value
        else:
            task.text = value

        data = self.new_prettify(project)
        with open(path_file, 'w') as file:
            file.write(data)


    def create_xml_file(self):
        project = ET.Element('project')

        name_project = ET.SubElement(project, 'name_project')
        name_project.text = 'First'

        task = ET.SubElement(project, 'task')
        task.text = 'Detection'

        model = ET.SubElement(project, 'model')
        model.text = 'yolov8'

        model_size = ET.SubElement(model, 'model_size')
        model_size.text = 'size'
        model_input = ET.SubElement(model, 'model_input')
        model_input.text = 'input'

        data = self.prettify(project)

        with open('file1.xml', 'w') as file:
            file.write(data)

    def prettify(self, elem):
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def new_prettify(self, elem):
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        xml_string = reparsed.toprettyxml()
        lines = [line for line in xml_string.split("\n") if line.strip()]
        return "\n".join(lines)

    def parse_xml_file(self):
        tree = ET.parse('file1.xml')
        root = tree.getroot()
        for elem in root:
            print(elem.text)
            for subelem in elem:
                #если есть текст и атрибуты
                print(subelem.text, subelem.attrib)

