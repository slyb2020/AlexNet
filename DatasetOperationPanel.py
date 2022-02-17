import cv2
import numpy as np
import wx
import wx.lib.scrolledpanel as scrolled

import images
from ID_DEFINE import *


class DatasetShowPanel(scrolled.ScrolledPanel):
    def __init__(self, parent, log, directory):
        self.log = log
        self.height = 50
        self.width = 50
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        wsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.buttonIdList = []
        self.buttonFilenameList = []
        for filename in os.listdir(directory)[:500]:
            img = cv2.imdecode(np.fromfile(trainDir + filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            id = wx.NewId()
            button = wx.Button(self, id, size=(self.width, self.height))
            a = img.shape[1]
            b = img.shape[0]
            bmp = wx.Image(img.shape[1], img.shape[0], img).Scale(width=self.width - 5, height=self.height - 5,
                                                                  quality=wx.IMAGE_QUALITY_NORMAL).ConvertToBitmap()
            button.SetBitmap(bmp)
            label = filename[:-4]
            button.SetToolTip(label)
            self.buttonIdList.append(id)
            self.buttonFilenameList.append(filename)
            wsizer.Add(button, 0)
        self.SetSizer(wsizer)
        self.SetAutoLayout(1)
        self.SetupScrolling()


class DatasetOperationPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)
        self.log = log
        self.datasetDir = None
        self.leftPanel = wx.Panel(self, -1, size=(300, -1))
        self.rightPanel = wx.Panel(self, -1, size=(300, -1))
        hbox = wx.BoxSizer()
        hbox.Add(self.leftPanel, 0, wx.EXPAND)
        hbox.Add(self.rightPanel, 1, wx.EXPAND)
        self.SetSizer(hbox)
        self.notebook = wx.Notebook(self.rightPanel, -1, size=(21, 21), style=
        # wx.BK_DEFAULT
        wx.BK_TOP
                                    # wx.BK_BOTTOM
                                    # wx.BK_LEFT
                                    # wx.BK_RIGHT
                                    # | wx.NB_MULTILINE
                                    )
        il = wx.ImageList(16, 16)
        il.Add(images._rt_smiley.GetBitmap())
        self.total_page_num = 0
        self.notebook.AssignImageList(il)
        idx2 = il.Add(images.GridBG.GetBitmap())
        idx3 = il.Add(images.Smiles.GetBitmap())
        idx4 = il.Add(images._rt_undo.GetBitmap())
        idx5 = il.Add(images._rt_save.GetBitmap())
        idx6 = il.Add(images._rt_redo.GetBitmap())
        hbox = wx.BoxSizer()
        self.trainSetPanel = DatasetShowPanel(self.notebook, self.log, trainDir)
        self.notebook.AddPage(self.trainSetPanel, "训练集")
        self.testSetPanel = wx.Panel(self.notebook)
        self.notebook.AddPage(self.testSetPanel, "测试集")
        hbox.Add(self.notebook, 1, wx.EXPAND)
        self.rightPanel.SetSizer(hbox)
        self.Bind(wx.EVT_BUTTON, self.OnPictureButton)
    def OnPictureButton(self, event):
        id = event.GetId()
        if id in self.trainSetPanel.buttonIdList:
            self.filename = self.trainSetPanel.buttonFilenameList[self.trainSetPanel.buttonIdList.index(id)]
            print("The filename is:", self.filename)
