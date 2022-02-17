import os
import wx
import images
import xml.etree.ElementTree as ET
import os
from ID_DEFINE import *


class AlexNetPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log
