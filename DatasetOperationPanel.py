import cv2
import numpy as np
import wx
import wx.lib.scrolledpanel as scrolled
import images
from ID_DEFINE import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch
import random
import tensorboard


class DoorTypeDataset(Dataset):
    def __init__(self, path, isTrain=True, size=None):
        """
        :param path: 数据集所处文件夹名
        :param isTrain: 是否是训练集数据
        :param isAug:   数据是否需要增广
        """
        self.imgPath = path + 'Training/'
        self.classes = [i.split('.')[0] for i in os.listdir(self.imgPath)]
        self.size = size
        self.filenames = []
        if isTrain:
            for dir in self.classes:
                for filename in os.listdir(self.imgPath + dir):
                    self.filenames.append(dir + '/' + filename)
        else:
            self.imgPath = path + 'Test/'
            for dir in self.classes:
                for filename in os.listdir(self.imgPath + dir):
                    self.filenames.append(dir + '/' + filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):  # Dataset类需要重写的方法，用于返回一个数据和标签对
        img = None
        img = cv2.imread(self.imgPath + self.filenames[index])  # 读取原始图像
        labelName = self.filenames[index].split('/')[0]
        h, w = img.shape[0:2]
        if self.size:
            padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
            if h > w:
                padw = (h - w) // 2
                img = np.pad(img, ((0, 0), (padw, padw), (0, 0)), 'constant', constant_values=0)
            elif w > h:
                padh = (w - h) // 2
                img = np.pad(img, ((padh, padh), (0, 0), (0, 0)), 'constant', constant_values=0)
            img = cv2.resize(img, (self.size, self.size))
        aug = transforms.Compose([
            transforms.ToTensor()
        ])
        img = aug(img)
        labels = [0] * len(self.classes)  # labels = self.filenames[index]
        idx = self.classes.index(labelName)
        labels[idx] = 1
        labels = torch.Tensor(labels)
        # labels = transforms.ToTensor()(labels) #感觉是因为labels不是picture所以不能使用ToTensor类来进行转换
        return img, labels


class PictureShowPanel(wx.Panel):
    def __init__(self, parent, log, size):
        wx.Panel.__init__(self, parent, -1, size=size)
        self.parent = parent
        self.log = log
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, evt):
        if self.parent.filename:
            dc = wx.PaintDC(self)
            self.img = cv2.imdecode(np.fromfile(trainDir + self.parent.filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.width, self.height = self.img.shape[1], self.img.shape[0]
            x, y = self.GetClientSize()
            bmp = wx.Image(self.width, self.height, self.img).Scale(width=x, height=y,
                                                        quality=wx.IMAGE_QUALITY_BOX_AVERAGE).ConvertToBitmap()
            dc.DrawBitmap(bmp, 0, 0, True)
        evt.Skip()


class DatasetButtonShowPanel(scrolled.ScrolledPanel):
    def __init__(self, parent, log, directory):
        self.log = log
        self.height = 50
        self.width = 50
        scrolled.ScrolledPanel.__init__(self, parent, -1)
        wsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        self.buttonIdList = []
        self.buttonFilenameList = []
        for filename in os.listdir(directory)[:500]:
            img = cv2.imread(trainDir + filename)
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
        self.leftPanel = PictureShowPanel(self, self.log, size=(300, -1))
        self.rightPanel = wx.Panel(self, -1, size=(300, -1))
        self.filename = None
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
        self.trainSetPanel = DatasetButtonShowPanel(self.notebook, self.log, trainDir)
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
            self.leftPanel.Refresh()
