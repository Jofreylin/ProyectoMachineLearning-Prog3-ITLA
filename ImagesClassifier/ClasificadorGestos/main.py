import eel
import testGesture
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
    result = testGesture.decodeIt(b64_string)
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

eel.start('view/index.html', size=(1000, 600))