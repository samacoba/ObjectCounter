#coding:utf-8
#１クラス検出　実行用スクリプト
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable
from chainer import optimizers
import numpy as np
import cupy as cp
import time
from PIL import Image
import json

#表示幅の変更
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:98% !important; }</style>"))

#標準スクリプト実行
with open('e_data.py', 'r', encoding='utf-8') as f:    
    exec(f.read()) #e_data.pyをスクリプト実行
with open('e_model.py', 'r', encoding='utf-8') as f:
    exec(f.read()) #e_model.pyをスクリプト実行

class Empty_structure:#構造体のように使う空のクラス 辞書型の[""]が書きずらく.でアクセスするため
    pass

Gv = Empty_structure()

#グローバル変数
Gv.B_out_ch = 3
Gv.BatchSize = 100
Gv.Recon_rate = 0.01
Gv.Peek_threshold = 0.2
Gv.Cir_color = ['R','G']

#モデル読み込み、初期設定
def run_set_model():

    gpu = 0 # 0：gpu使用、-1：gpu不使用

    global Md#モデル
    global xp
    global Op#オプティマイザ  
    global Iteration
    Iteration = 0

    Md ={}
    Op ={}

    xp = cuda.cupy
    cuda.get_device(gpu).use()

    
    Md['g_ab_c'] = Conv()
    Md['g_ab_d'] = DeConv(out_ch = Gv.B_out_ch)

    Md['g_ba_c'] = Conv()
    Md['g_ba_d'] = DeConv(out_ch = 3)


    Md['d_a'] = Discriminator()
    Md['d_b'] = Discriminator()


    for key in Md.keys():
        Md[key].to_gpu()
        Op[key] = optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
        Op[key].setup(Md[key])
        Op[key].add_hook(chainer.optimizer.WeightDecay(1e-4))  

    

#トレーニングとテスト
def run_train(N_iteration = 1000, test_interval = 20):

    global Iteration    

    for loop in range(N_iteration): 

        run_disco_update()         
        Iteration = Iteration + 1

        if (loop % test_interval) ==  (test_interval - 1):#インターバルの最後にテスト
            run_test()

    print(len(DataA['posi']),'\t', '学習終了')


#DiscoGAN UPdate
def run_disco_update():

    # read data    
    DataN = get_data_N_rand(DataA, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])
    x_a = Variable(xp.asarray(DataN['x'])) 
    
    DataN = get_data_N_rand(DataB, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])        
    x_b = Variable(xp.asarray(DataN['x']))


    # conversion
    x_ab = Md['g_ab_d'](Md['g_ab_c'](x_a))
    x_ba = Md['g_ba_d'](Md['g_ba_c'](x_b))

    # reconversion
    x_aba = Md['g_ba_d'](Md['g_ba_c'](x_ab))
    x_bab = Md['g_ab_d'](Md['g_ab_c'](x_ba))

    # reconstruction loss
    recon_loss_a = F.mean_squared_error(x_a, x_aba)
    recon_loss_b = F.mean_squared_error(x_b, x_bab)

    # discriminate
    y_a_real, feats_a_real = Md['d_a'](x_a)
    y_a_fake, feats_a_fake = Md['d_a'](x_ba)

    y_b_real, feats_b_real = Md['d_b'](x_b)
    y_b_fake, feats_b_fake = Md['d_b'](x_ab)

    # reconstruction loss
    recon_loss_a = F.mean_squared_error(x_a, x_aba)
    recon_loss_b = F.mean_squared_error(x_b, x_bab)

    # discriminate
    y_a_real, feats_a_real = Md['d_a'](x_a)
    y_a_fake, feats_a_fake = Md['d_a'](x_ba)

    y_b_real, feats_b_real = Md['d_b'](x_b)
    y_b_fake, feats_b_fake = Md['d_b'](x_ab)  


    # GAN loss
    gan_loss_dis_a, gan_loss_gen_a = compute_loss_gan(y_a_real, y_a_fake)
    feat_loss_a = compute_loss_feat(feats_a_real, feats_a_fake)

    gan_loss_dis_b, gan_loss_gen_b  = compute_loss_gan(y_b_real, y_b_fake)
    feat_loss_b = compute_loss_feat(feats_b_real, feats_b_fake)
   

    # compute loss
    total_loss_gen_a = (1.-Gv.Recon_rate)*(0.1*gan_loss_gen_b + 0.9*feat_loss_b) + Gv.Recon_rate * recon_loss_a
    total_loss_gen_b = (1.-Gv.Recon_rate)*(0.1*gan_loss_gen_a + 0.9*feat_loss_a) + Gv.Recon_rate * recon_loss_b

    gen_loss = total_loss_gen_a + total_loss_gen_b 
    dis_loss = gan_loss_dis_a + gan_loss_dis_b 

    if Iteration % 3 == 0:
        Md['d_a'].cleargrads()
        Md['d_b'].cleargrads()
        dis_loss.backward()
        Op['d_a'].update()
        Op['d_b'].update()
    else:
        Md['g_ab_c'].cleargrads()
        Md['g_ab_d'].cleargrads()
        Md['g_ba_c'].cleargrads()
        Md['g_ba_d'].cleargrads()
        gen_loss.backward()
        Op['g_ab_c'].update()
        Op['g_ab_d'].update()
        Op['g_ba_c'].update()
        Op['g_ba_d'].update()

def compute_loss_gan(y_real, y_fake):
    batchsize = y_real.shape[0]
    loss_dis = 0.5 * F.sum(F.softplus(-y_real) + F.softplus(y_fake))
    loss_gen = F.sum(F.softplus(-y_fake))
    return loss_dis / batchsize, loss_gen / batchsize

def compute_loss_feat(feats_real, feats_fake):
    losses = 0
    for feat_real, feat_fake in zip(feats_real, feats_fake):
        feat_real_mean = F.sum(feat_real, 0) / feat_real.shape[0]
        feat_fake_mean = F.sum(feat_fake, 0) / feat_fake.shape[0]
        l2 = (feat_real_mean - feat_fake_mean) ** 2
        loss = F.sum(l2) / l2.size
        losses += loss
    return losses

def compute_entro_X(x):
    #エントロピー的なものを計算
    delta = chainer.Variable(1e-6 * cp.ones((x.shape[0], 1, x.shape[2], x.shape[3]), dtype = np.float32 ))

    #ch平均
    x_m = F.relu(F.mean(x, axis=1, keepdims = True))
    #ノーマライズ 
    x_n = x_m / (F.sum(x_m) / x_m.shape[0] + delta) * 100
    #0-1にクリップ
    x_n = F.clip(x_n, 0, 1)
    #ピクセルごとにエントロピー*を計算
    entro = x_n * F.log2( 1/(x_n  + delta)) + (1 - x_n) * F.log2( 1/(1 - x_n  + delta)) 
    #ピクセルのエントロピーを合計
    entro = F.sum(entro)

    return cp.asnumpy(entro.data) # ,cp.asnumpy(x_n.data)*255

   
#＜テスト部分スクリプト＞            
def run_test():

    x_a = chainer.Variable(xp.asarray(DataA['x']))
    x_ax = Md['g_ab_c'](x_a)
    x_ab = Md['g_ab_d'](x_ax)
    DataA['y'] = cp.asnumpy(x_ab.data)

    #中間のエントロピー的なものを計算
    #entro_X = compute_entro_X(x_ax)

    #中間部分を可視化
    #DataA['m'] = np.mean(cp.asnumpy(x_ax.data),axis=1, keepdims = True) * 100
    

    #BtoAを計算            
    # x_b = chainer.Variable(xp.asarray(DataB['x'][0:1]))
    # x_ba = Md['g_ba_d'](Md['g_ba_c'](x_b))
    # DataB['y'] = cp.asnumpy(x_ba.data)


    #ピーク位置検出
    DataA['posi'] = get_posi_from_img(DataA['y'], threshold = Gv.Peek_threshold )

    #円重ね描き
    DataA['x_y_circle'] = add_circle_from_posi(DataA['x'], DataA['posi'], color = Gv.Cir_color[0])


    #bokeh描画
    imgs = [DataA['x_y_circle'][0], DataA['y'][0]]
    infos = ['iteration = ' + str(Iteration), 'count = ' + str(len(DataA['posi'])) ]
    bokeh_update_img(imgs = imgs , infos = infos)
    





