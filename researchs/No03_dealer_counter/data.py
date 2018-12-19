# coding: utf-8
# データ部分

import chainer
import chainer.links as L
import chainer.functions as F


from skimage import draw
from skimage import transform

import numpy as np
from numpy import random

import math
import pickle

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import push_notebook, show, output_notebook

from PIL import Image



Img_Scale = 255 #画像のスケール 1⇒(0-1) , 255⇒(0-255)

#Bokehで表示用
class Bokeh_View:

    def __init__(self, imgs):#初期表示【imgs = [3ch][H][W](0-1 or 0-255 fp)】
        
        _, imgH, imgW = imgs[0].shape
        
        img1 = get_view_img(imgs[0])
        img2 = get_view_img(imgs[1])

        self.plt1 = figure(title = 'train N = --', x_range=[0, imgW], y_range=[0, imgH])
        self.rend1 = self.plt1.image_rgba(image=[img1],x=[0], y=[0], dw=[imgW], dh=[imgH])

        self.plt2 = figure(title = 'count  = --', x_range=self.plt1.x_range, y_range=self.plt1.y_range)
        self.rend2 = self.plt2.image_rgba(image=[img2],x=[0], y=[0], dw=[imgW], dh=[imgH])

        prob =[0.25, 0.25, 0.25, 0.25]
        self.plt3 = figure(x_range= ['13個', '26個', '52個', '131個'], y_range=[0,1], title="ディーラ確率分布")
        self.rend3 = self.plt3.vbar(x=['13個', '26個', '52個', '131個'], top=prob, width=0.8)

        self.plts = gridplot([[self.plt1,self.plt2,self.plt3]], plot_width = 300, plot_height = 300)        

        output_notebook()

        self.handle = show(self.plts, notebook_handle=True)

    def update_img(self, imgs, prob, trainN = 0, count  = 0):#アップデート表示【imags = [3ch][H][W](0-1 or 0-255 fp)】

        img1 = get_view_img(imgs[0])
        img2 = get_view_img(imgs[1])

        self.rend1.data_source.data['image'] = [img1]
        self.rend2.data_source.data['image'] = [img2]
        self.rend3.data_source.data['top'] = prob

        self.plt1.title.text='train N = '+ str(trainN)
        self.plt2.title.text='count  =  '+ str(count)

        push_notebook(handle = self.handle)
    
    #画像で保存
    def save_img(self,fname = 'out.png'):
        from bokeh.io import export_png
        export_png(self.plts, filename = fname)
           
   
#bokeh用のイメージを取得img = [3][imgH][imgW](0-1 or 0-255 fp)
def get_view_img(img):
 
    ch, imgW, imgH= img.shape
    if ch == 1:#1chの場合 ⇒ 3ch
        img = np.broadcast_to(img, (3, imgH, imgW))
    img = img * (255/Img_Scale) #0-1スケールの時⇒0-255へ
    img = np.clip(img, 0, 255).transpose((1, 2, 0))
    img_plt = np.empty((imgH,imgW), dtype="uint32")
    view = img_plt.view(dtype=np.uint8).reshape((imgH, imgW, 4))
    view[:, :, 0:3] = np.flipud(img[:, :, 0:3])#上下反転あり
    view[:, :, 3] = 255    
    return img_plt

#画像をRGB化、センター加工、リサイズ
def cRBG_squre_resize(img, out_size = 512):#imgはpil Image
   
    img = img.convert('RGB')    
    w,h = img.size
    if w > h :
        box = ((w-h)//2, 0, (w-h)//2 + h, h)
    elif h > w :
        box = (0, (h-w)//2, w, (h-w)//2 + w)     
    else :
        box = (0, 0, w, h)        
    img = img.resize((out_size, out_size), box = box)    
    return img


#ファイル ⇒ [1][ch][imgH][imgW] fp32(0-1 or 0-255) を作成
def load_img(fpath = 'imgA1.png'):    
    img = np.asarray(Image.open(fpath))
    img = img.astype("float32").transpose((2, 0, 1))/(255/Img_Scale)#(0-1 or 0-255)
    img = img.reshape((1,*img.shape))
    return img

def resize_img(fpath = 'imgA1.png'):
    img = Image.open(fpath)
    img = cRBG_squre_resize(img, out_size = 512)#RGB化、center切り取り、リサイズ
    img.save('out.png', format ='PNG')
    img = load_img(fpath ='out.png')
    return img


#極大値を取得、in_imgs = ndarray[N][1][imgH][imgW]
def get_local_max_point(in_imgs, threshold = 0.2): #threshold は0-1にて範囲指定
    threshold =  Img_Scale * threshold
    m_imgs = chainer.Variable(in_imgs)
    m_imgs = F.max_pooling_2d(m_imgs, ksize=9 ,stride=1, pad=4)
    m_imgs = m_imgs.data
    p_array = (in_imgs == m_imgs) #極大値判定（True or False）の配列
    out_imgs = in_imgs * p_array
    
    out_imgs[out_imgs >= threshold] = Img_Scale
    out_imgs[out_imgs < threshold] = 0
    
    return out_imgs/Img_Scale #pointは0/1のみで返却

def get_posi_from_point(imgp):#point の [1][1][imgH][imgW] ⇒ posi
    
    _, imgW, imgH= imgp[0].shape
    posi =[]
    for x in range(imgW):
        for y in range(imgH):
            if imgp[0][0][y][x]:
                posi.append({'x':x, 'y':y})

    return posi


#データの切り出し        
def get_data_N_rand(DataO, N_pic =1, imgH = 256, imgW = 256, keys =['x','t_core']):
  
    Data={}
    
    #切り出したデータの保存先 dim=[N][1][imgH][imgW] ,float32       
    for key in keys:
        Data[key] = np.zeros((N_pic, DataO[key].shape[1], imgH, imgW), dtype = "float32")
    
    #切り出し限界を設定
    xlim = DataO[keys[0]].shape[3] - imgW + 1
    ylim = DataO[keys[0]].shape[2] - imgH + 1


    im_num =np.random.randint(0, DataO[keys[0]].shape[0], size=N_pic)#切り取る写真の番号
    rotNo = np.random.randint(4, size=N_pic) #回転No
    flipNo = np.random.randint(2, size=N_pic) #フリップNo
    cutx = np.random.randint(0, xlim, size=N_pic)
    cuty = np.random.randint(0, ylim, size=N_pic)

    for i in range(0, N_pic):          
        for key in keys:
            Data[key][i] = rand_rot((DataO[key][im_num[i]][:, cuty[i]:cuty[i]+imgH, cutx[i]:cutx[i]+imgW]), rotNo[i], flipNo[i])
    
    return Data 

#np配列をもらって左右上下の反転・90、180、270°の回転した配列を返す
def rand_rot(img, rotNo, flipNo):#img[ch][H][W]
    img = np.rot90(img, k=rotNo, axes=(1,2))
    if flipNo:
        img = np.flip(img, axis=2)
    return img 

                


#circleを描画 in_imgs = ndarray[1][1][imgH][imgW]
def draw_circle(in_imgs):
        
    cir = np.zeros((1,1,15,15), dtype= "float32")
    rr, cc = draw.circle_perimeter(7,7,5)
    cir[0][0][rr, cc] = Img_Scale #0-1 or 0-255      
    decon_cir = L.Deconvolution2D(1, 1, 15, stride=1, pad=7)
    decon_cir.W.data = cir
    out_imgs  = decon_cir(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs

#コアを描画、in_imgs = ndarray[1][1][imgH][imgW]
def draw_core(in_imgs, sig=3.0, max_xy = 15, c_xy= 7):
    
    sig2=sig*sig
    #c_xy=7
    core=np.zeros((max_xy, max_xy),dtype = "float32")
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1,1,core.shape[0],core.shape[1]))
    
    decon_core = L.Deconvolution2D(1, 1, max_xy, stride=1, pad=c_xy)
    decon_core.W.data = core
    out_imgs  = decon_core(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs


#点と球状のimageを作成
def get_rand_core(N_pic=1, imgH=512, imgW=512, threshold = 0.9995, ch = 3):


    #ランダムに0.05％の点を作る
    img_p = np.random.rand(N_pic*imgW*imgH)
    img_p[img_p < threshold] = 0
    img_p[img_p >= threshold] = 1

    img_p = img_p.reshape((N_pic,1, imgH, imgW)).astype("float32")

    #点⇒球に変換
    img = draw_core(img_p) *Img_Scale    
    
    if ch ==3 :#ch1⇒ch3
        img = np.broadcast_to(img, (N_pic, 3, imgH, imgW))
    
    return img

#posi ⇒ core 変換
def set_core_img(imgx, posi, ch = 1):#imgx = [1][3ch][H][W] posi= [{x:a,y:b},…]
    
    #'X'の画像からサイズ抽出
    imgH, imgW = imgx[0][0].shape
    img_p = np.zeros((1, 1, imgH, imgW), dtype= "float32")
    for p in posi:
        try:
            img_p[0][0][p['y']][p['x']] = 1
        except:
            print('範囲外入力エラー？')

    #点⇒球に変換
    img = draw_core(img_p)* Img_Scale

    if ch ==3 :#ch1⇒ch3
        img = np.broadcast_to(img, (1, 3, imgH, imgW))

    return img #[1][1ch/3ch][H][W]


#周辺のみの部分的なXイメージの作成
def set_part_img(imgx, posi, ch=3):
    imgH, imgW = imgx[0][0].shape
    img_p = np.zeros((1, 1, imgH, imgW), dtype= "float32")
    for p in posi:
        try:
            img_p[0][0][p['y']][p['x']] = 1 #ここはImg_Scale不要
        except:
            print('範囲外入力エラー？')
            
    img = draw_core(img_p, sig=20,c_xy =40,max_xy = 81)
    img = np.clip(img, 0, 1)
    img = np.broadcast_to(img, (1, 3, imgH, imgW))

    img_av = np.broadcast_to(imgx.mean(), (1, 3, imgH, imgW))#平均画像
    
    img_part = imgx*img + img_av*(1-img)
    
    return img_part


    
 


