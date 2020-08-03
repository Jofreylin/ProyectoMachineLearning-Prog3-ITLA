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
def gesture(filepath):
    ip = TestModel2.test(filepath)
    return ip

@eel.expose
def btn_ResimyoluClick():
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askopenfilename()
    return folder

eel.start('view/index.html', size=(1000, 600),mode='edge')