import eel
import TestModel2
import faceTrain
import faceTest
import captureFaces
import faceRecog
from tkinter import *
from tkinter import filedialog

"""
-------------------------
INSTALACIONES
-------------------------
pip install eel
=========================
"""

eel.init('files_dependencies/web')

@eel.expose
def getGesture(b64_string):
    result = TestModel2.decodeIt(b64_string)
    return result

@eel.expose
def detectFaces(b64_string):
    result = faceRecog.decodeIt(b64_string)
    return result

@eel.expose
def detectCameraFaces():
    result = faceTest.test()
    return result

@eel.expose
def setFaces(nombre,cantidad):
    result = captureFaces.capture(nombre,cantidad)
    return result

@eel.expose
def trainFace():
    result = faceTrain.train()
    return result

@eel.expose
def searchFace(nombre):
    result = captureFaces.searchClass(nombre)
    return result

@eel.expose
def btn_ResimyoluClick():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askopenfilename()
    return folder



eel.start('view/index.html', size=(1000, 600))