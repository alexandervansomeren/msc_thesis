import glob
import xml.etree.ElementTree as ET
import lxml.etree as etree


def parse_citeseerx_xml(xml_file):
    tree = ET.parse(xml_file)
    print(etree.tostring(tree, pretty_print=True))
    title = tree.find('title').text
    authors = [author.text for author in tree.findall('author')]


for xml_file in glob.glob('../../data/citeseer_2017.tmp/citeseerx-partial-papers/*.xml'):
    parse_citeseerx_xml(xml_file)
    break
