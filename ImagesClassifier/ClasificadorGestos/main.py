import eel
import TestModel2
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
def gestureRecognition(b64_string):
    result = TestModel2.decodeIt(b64_string)
    return result

@eel.expose
def btn_ResimyoluClick():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askopenfilename()
    return folder

eel.start('view/index.html', size=(1000, 600))