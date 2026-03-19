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



def writemd(im, #root from xml.etree.ElementTree tree
            file_name: str,
            ):
    root = getmdroot(im)
    st = ET.tostring(root, encoding='utf8')
    utf8_string = st.decode('utf8')
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(utf8_string)


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def getstarttime(im, #root from xml.etree.ElementTree tree
                 ):
    root = getmdroot(im)
    adt = root.find('Information/Image/AcquisitionDateAndTime').text
    dt = re.findall(r'T(.*?)\.',adt)[0]
    return get_sec(dt)

def gettimeinterval(im, #root from xml.etree.ElementTree tree
                    ):
    root = getmdroot(im)
    return float(root.find('Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/TimeSeriesSetup/Interval/TimeSpan/Value').text)

def frame_range_in_subset(im,#root from xml.etree.ElementTree tree
                   ):
    root = getmdroot(im)
    subsetstring = root.find('CustomAttributes/SubsetString').text
    if 'T' in subsetstring:
        first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
    else:
        first = 1
        last = int(root.find('Information').find('Image').find('SizeT').text)
    return first, last  

def framesinsubset(im,#root from xml.etree.ElementTree tree
                   ):
    root = getmdroot(im)
    subsetstring = root.find('CustomAttributes/SubsetString').text
    if 'T' in subsetstring:
        first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
        framenum = int(last-first+1)
    else:
        framenum = int(root.find('Information').find('Image').find('SizeT').text)
    return framenum
        
        
def adjustedstarttime(im, #root from xml.etree.ElementTree tree
                      ):
    root = getmdroot(im)
    starttime = getstarttime(im)
    subsetstring = root.find('CustomAttributes/SubsetString').text
    time_interval = gettimeinterval(im)
    if 'T' in subsetstring:
        first, last = [int(x) for x in re.findall(r'T\(([^)]*)\)',subsetstring)[0].split('-')]
        if first != 1:
            starttime = starttime + (first-1)*time_interval
    return starttime


def getscale(im, #root from xml.etree.ElementTree tree
             ):
    
    ### returns scale in microns in xyz order
    root = getmdroot(im)
    # Unpack short info scales
    list_xs = root.findall(".//Distance[@Id='X']")
    list_ys = root.findall(".//Distance[@Id='Y']")
    list_zs = root.findall(".//Distance[@Id='Z']")
    scale_xe = list_xs[0].find("./Value")
    scale_ye = list_ys[0].find("./Value")
    scale_ze = None if len(list_zs) == 0 else list_zs[0].find("./Value")

    # Unpack the string value to a float
    # Split by "E" and take the first part because the values are stored
    # with E-06 for micrometers, even though the unit is also present in metadata
    # 🤷
    if scale_xe is not None and scale_xe.text is not None:
        scale_x = float(scale_xe.text.split("E")[0])/(int(10**abs(float(scale_xe.text.split("E")[1])))/1000000)
    if scale_ye is not None and scale_ye.text is not None:
        scale_y = float(scale_ye.text.split("E")[0])/(int(10**abs(float(scale_xe.text.split("E")[1])))/1000000)
    if scale_ze is not None and scale_ze.text is not None:
        scale_z = float(scale_ze.text.split("E")[0])/(int(10**abs(float(scale_xe.text.split("E")[1])))/1000000)
    return scale_x, scale_y, scale_z



