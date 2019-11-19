# coding: utf-8
# データ用スクリプト
# 外部依存がなく、関数内で完結する関数が中心
import chainer
import chainer.links as L
import chainer.functions as F

from skimage import draw
from skimage import transform

import numpy as np
import cupy as cp

import math
import pickle

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import push_notebook, output_notebook

from PIL import Image

### Bokeh関連 ###

def get_bokeh_view(imgs, plot_w =512, plot_h =512):# 
    """ imgs = [[1/3ch][H][W], ……] (0-255 fp)(np or cp))　
        return : Bv (Bokehのviewを扱うグローバル辞書オブジェクト) """
    
    _, imgH, imgW = imgs[0].shape   

    Bv = {}
    Bv['p'] = []
     
    for i in range(len(imgs)):
        
        Bv['p'].append({})
        if i == 0:#1枚目
            Bv['p'][i]['fig'] = figure(title = '', x_range=[0, imgW], y_range=[imgH, 0])#y_range=[imgH, 0]によりy軸を反転
        else :#2枚以上の場合
            Bv['p'][i]['fig'] = figure(title = '', x_range=Bv['p'][0]['fig'].x_range, y_range=Bv['p'][0]['fig'].y_range)

        v_img = bokeh_conv_view_img(imgs[i])#[1/3ch][H][W] ⇒ bokehのイメージに変換

        Bv['p'][i]['img']= Bv['p'][i]['fig'].image_rgba(image=[v_img],x=[0], y=[imgH], dw=[imgW], dh=[imgH])#反転軸のためy=[imgH]

    gplots = [Bv['p'][i]['fig'] for i in range( len(imgs))] 
    Bv['gp'] = gridplot( [gplots], plot_width = plot_w, plot_height = plot_h)

    output_notebook()
    from bokeh.io import show
    Bv['handle'] = show(Bv['gp'], notebook_handle=True)

    return Bv


def bokeh_conv_view_img(img):
    """ [1/3ch][H][W] 0-255⇒ bokehのイメージに変換 """
    if type(img) == cp.core.core.ndarray:#cupy配列の場合
        img =cp.asnumpy(img)

    ch, imgW, imgH= img.shape
    if ch == 1:#1chの場合 ⇒ 3ch
        img = np.broadcast_to(img, (3, imgH, imgW))
    img = np.clip(img, 0, 255).transpose((1, 2, 0))
    img_plt = np.empty((imgH,imgW), dtype="uint32")
    view = img_plt.view(dtype=np.uint8).reshape((imgH, imgW, 4))
    view[:, :, 0:3] = np.flipud(img[:, :, 0:3])#上下反転あり
    view[:, :, 3] = 255    
    return img_plt

def bokeh_update_img(imgs, infos  = []):#
    """ アップデート表示 imgs = [[1/3ch][H][W], ……] (0-1 or 0-255 fp)(np or cp) """

    for i in range(len(imgs)):
        v_img = bokeh_conv_view_img(imgs[i])
        Bv['p'][i]['img'].data_source.data['image'] = [v_img]
    
    for i in range(len(infos)):
        Bv['p'][i]['fig'].title.text= infos[i]

    push_notebook(handle = Bv['handle'])


def bokeh_save_img(fname = 'out.png'):
    """bokeh画像を保存"""
    from bokeh.io import export_png
    export_png(Bv['gp'], filename = fname)  



### 画像ファイル関係 ###

def load_img(fpath = 'imgA1.png'): 
    """ 画像ファイルパス ⇒ [1][ch][imgH][imgW] (0-255 fp32) を返却 """   
    img = np.asarray(Image.open(fpath))
    img = img.astype("float32").transpose((2, 0, 1))
    img = img.reshape((1,*img.shape))
    return img

def save_png_img(img, fpath = 'result.png'):
    """PNG保存 img =[1][1 or 3ch][imgH][imgW] (0-255 fp32) ,fpath = 画像ファイルパス"""
    
    if img.shape[1] == 1:#1ch
        img = img[0][0]        
    elif img.shape[1] == 3:#3ch
        img = img[0].transpose(1,2,0)
    else :
        print('入力chエラー')
        return

    img = (np.clip(img,0,255)).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(fpath, format ='PNG')


def load_resize_img(fpath = 'imgA1.png', opath = 'resize.png', out_size = 512):
    """ 画像ファイルパス ⇒ センターリサイズ ⇒ 一時保存 ⇒[1][ch][imgH][imgW] (0-1 or 0-255, fp32) を返却 """

    #画像をRGB化、センター加工、リサイズ
    def conv_RBG_squre_resize(img, out_size = 512):#imgはpil Image    
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

    img = Image.open(fpath)
    img = conv_RBG_squre_resize(img, out_size = out_size)#RGB化、center切り取り、リサイズ
    img.save(opath, format ='PNG')#一時保存(imgはpilのままなので)
    img = load_img(fpath = opath )#画像ファイルを[1][ch][imgH][imgW]で読み出し
    return img


### データ加工関連 ###

      
def get_data_N_rand(DataO, N_pic =1, imgH = 256, imgW = 256, keys =['x','t_core']):
    """ ランダムでデータの切り出し 入力：Data ⇒ return: 切り出し後の新たなDataを返却 """
    Data={}
    
    #切り出したデータの保存先 dim=[N][ch][imgH][imgW] ,float32       
    for key in keys:
        Data[key] = np.zeros((N_pic, DataO[key].shape[1], imgH, imgW), dtype = "float32")
    
    #切り出し限界を設定
    xlim = DataO[keys[0]].shape[3] - imgW + 1
    ylim = DataO[keys[0]].shape[2] - imgH + 1

    im_num =np.random.randint(0, DataO[keys[0]].shape[0], size=N_pic)#複数枚の内、切り取る写真の番号
    rotNo = np.random.randint(4, size=N_pic) #回転No
    flipNo = np.random.randint(2, size=N_pic) #フリップNo
    cutx = np.random.randint(0, xlim, size=N_pic)
    cuty = np.random.randint(0, ylim, size=N_pic)

    #np配列をもらって左右上下の反転・90、180、270°の回転した配列を返す
    def rand_rot(img, rotNo, flipNo):#img[ch][H][W]
        img = np.rot90(img, k=rotNo, axes=(1,2))
        if flipNo:
            img = np.flip(img, axis=2)
        return img 

    for i in range(0, N_pic):          
        for key in keys:
            Data[key][i] = rand_rot((DataO[key][im_num[i]][:, cuty[i]:cuty[i]+imgH, cutx[i]:cutx[i]+imgW]), rotNo[i], flipNo[i])
    
    return Data 
             

def get_local_max_point(in_imgs, threshold = 0.2): 
    """ 極大値を取得、in_imgs = [N_pic][1][imgH][imgW] np, fp32 (0-1/0-255), threshold は0-1にて範囲指定 
        ⇒return :[N_pic][1][imgH][imgW] np, fp32 (0-1)"""

    threshold =  255 * threshold
    m_imgs = chainer.Variable(in_imgs)
    m_imgs = F.max_pooling_2d(m_imgs, ksize=9 ,stride=1, pad=4)
    m_imgs = m_imgs.data
    p_array = (in_imgs == m_imgs) #極大値判定（True or False）の配列
    out_imgs = in_imgs * p_array
    
    out_imgs[out_imgs >= threshold] = 255
    out_imgs[out_imgs < threshold] = 0
    
    return out_imgs/255 #pointは0/1のみで返却



def conv_point_to_posi(img_p):
    """ 点画像 [1][1][imgH][imgW] ⇒　点位置 posi[N][y,x] """  

    posi = np.where(img_p[0][0] == 1)    
    posi = np.asarray(posi)#1列アレイが２本のリストとして出てくる
    posi = posi.transpose(1,0).tolist()    
    
    return posi



def conv_point_to_circle(in_imgs):
    """ 点画像 ⇒ circleを描画、in_imgs = [N_pic][1][imgH][imgW] np, fp32 (0-1)
        ⇒return: [N_pic][1][imgH][imgW] np, fp32 0-255"""
    cir = np.zeros((1,1,15,15), dtype= "float32")
    rr, cc = draw.circle_perimeter(7,7,5)
    cir[0][0][rr, cc] = 255
    out_imgs = F.deconvolution_2d(in_imgs, W = cir, b = None, stride = 1, pad = 7)
    return out_imgs.data


def conv_point_to_core(in_imgs, sig=3.0, max_xy = 15, c_xy= 7):
    """ 点画像 ⇒ コアを描画、in_imgs = [1][1][imgH][imgW] np, fp32 (0-1)
        ⇒return: [N_pic][1][imgH][imgW] np, fp32 (0-1)"""
    sig2=sig*sig
    core=np.zeros((max_xy, max_xy), dtype = "float32")
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1, 1, core.shape[0],core.shape[1]))    
    out_imgs = F.deconvolution_2d(in_imgs, W = core, b = None, stride = 1, pad=c_xy)

    return out_imgs.data


def get_rand_core(N_pic=1, imgH=512, imgW=512, p_num = 100, ch = 3, sig=3.0, max_xy = 15, c_xy= 7):
    """ ランダムガウス玉を取得　⇒return: [N_pic][ch][imgH][imgW] np, fp32(0-1/0-255)"""
    
    threshold = 1-float(p_num)/(imgW*imgH)#p_numの粒子数になるように閾値を設定
    img_p = np.random.rand(N_pic*imgW*imgH)
    img_p[img_p < threshold] = 0
    img_p[img_p >= threshold] = 1

    img_p = img_p.reshape((N_pic,1, imgH, imgW)).astype("float32")

    #点⇒球に変換
    img = conv_point_to_core(img_p, sig = sig, max_xy = max_xy, c_xy = c_xy) *255  
    
    if ch ==3 :#ch1⇒ch3
        img = np.broadcast_to(img, (N_pic, 3, imgH, imgW))
    
    return img


def get_posi_from_img(img, threshold = 0.2):
    """画像 img = [1][3 or 1][imgH][imgW] ⇒ 点位置 posi = [N][y,x] """  
    img_p = get_local_max_point(F.mean(img, axis = 1, keepdims = True).data, threshold = threshold)
    posi = conv_point_to_posi(img_p)
    return posi


def add_circle_from_posi(img, posi, color = 'R'): 
    """ 円が追加される画像 img = [1][3][imgH][imgW] ,点位置 posi = [N][y,x],
    　  color = 'R','G','B','C','M','Y','W',その他は黒,'N'は何もしない
        ⇒ return 円が追加された画像 [1][3][imgH][imgW]"""

    if color == 'N':#Nの時はそのまま返却
        return img

    imgH, imgW = img[0][0].shape    
    img = img.copy()

    for i in range(len(posi)):
        
            #円の塗り替えるピクセルのインデックスを取得
            rr, cc = draw.circle_perimeter(posi[i][0],posi[i][1], radius = 5, shape = (imgH, imgW)) 
            
            if color in('R', 'M' ,'Y', 'W'):
                img[0][0][rr, cc] = 255 #赤〇追加 
            else:
                img[0][0][rr, cc] = 0
            if color in('G', 'Y', 'C', 'W'):    
                img[0][1][rr, cc] = 255 #緑〇追加
            else:
                img[0][1][rr, cc] = 0             
            if color in('B', 'C', 'M', 'W') :    
                img[0][2][rr, cc] = 255 #青〇追加
            else:
                img[0][2][rr, cc] = 0 
                        
    return img




    
 


