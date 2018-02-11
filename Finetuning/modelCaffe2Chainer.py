from chainer.serializers import npz
import os,tkinter.filedialog,tkinter,tkinter.messagebox
from chainer.links.caffe import CaffeFunction
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
path = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
model = CaffeFunction(path)
npz.save_npz('vgg16.npz', model, compression=False)
