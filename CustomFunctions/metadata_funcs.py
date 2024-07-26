# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:58:48 2024

@author: Aaron
"""

import re
import xml.etree.ElementTree as ET



def getmdroot(im, #image from aicsimageio CziReader
              ):
    tree = ET.ElementTree(im.metadata[0])
    return tree.getroot()



def writemd(root, #root from xml.etree.ElementTree tree
            file_name: str,
            ):
    st = ET.tostring(root, encoding='utf8')
    utf8_string = st.decode('utf8')
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(utf8_string)


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def getstarttime(root, #root from xml.etree.ElementTree tree
                 ):
    adt = root.find('Information').find('Image').find('AcquisitionDateAndTime').text
    dt = re.findall(r'T(.*?)\.',adt)[0]
    return get_sec(dt)

def gettimeinterval(root, #root from xml.etree.ElementTree tree
                    ):
    return float(root.find('Experiment').find('ExperimentBlocks').find('AcquisitionBlock').find('SubDimensionSetups').find('TimeSeriesSetup').find('Interval').find('TimeSpan').find('Value').text)

def framesinsubset(root,#root from xml.etree.ElementTree tree
                   ):
    subsetstring = root.find('CustomAttributes').find('SubsetString').text
    if 'T' in subsetstring:
        first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
        framenum = int(last-first+1)
    else:
        framenum = int(root.find('Information').find('Image').find('SizeT').text)
    return framenum
        
        
def adjustedstarttime(root, #root from xml.etree.ElementTree tree
                      ):
    starttime = getstarttime(root)
    subsetstring = root.find('CustomAttributes').find('SubsetString').text
    time_interval = gettimeinterval(root)
    if 'T' in subsetstring:
        first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
        if first != 1:
            starttime = starttime + (first-1)*time_interval
    return starttime