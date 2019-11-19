#coding:utf-8
#２クラス検出実行用スクリプト
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

    
    Md['g_ab_c1'] = Conv()
    Md['g_ab_c2'] = Conv()
    Md['g_ab_d'] = DeConv(out_ch = Gv.B_out_ch)

    Md['g_ba_c'] = Conv()
    Md['g_ba_d1'] = DeConv(out_ch = 3)
    Md['g_ba_d2'] = DeConv(out_ch = 3)

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

    print(len(DataA['posi1']),'\t', len(DataA['posi2']),'\t','学習終了')


#DiscoGAN UPdate
def run_disco_update():

    # read data    
    DataN = get_data_N_rand(DataA, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])
    x_a = Variable(xp.asarray(DataN['x'])) 
    
    DataN = get_data_N_rand(DataB, N_pic = Gv.BatchSize, imgH = 64, imgW = 64, keys =['x'])        
    x_b = Variable(xp.asarray(DataN['x']))


    # conversion
    x_ax_1 = Md['g_ab_c1'](x_a)
    x_ax_2 = Md['g_ab_c2'](x_a)
    x_ab_1 = Md['g_ab_d'](x_ax_1)
    x_ab_2 = Md['g_ab_d'](x_ax_2)

    x_ab = F.concat((x_ab_1, x_ab_2), axis=1)#3ch,3ch⇒6ch

    x_b_1, x_b_2 = F.split_axis(x_b, 2, axis=1)#6ch⇒3ch,3ch
    x_ba_1 = Md['g_ba_d1'](Md['g_ba_c'](x_b_1))
    x_ba_2 = Md['g_ba_d2'](Md['g_ba_c'](x_b_2))

    x_ba = x_ba_1 + x_ba_2

    # reconversion
    x_aba_1 = Md['g_ba_d1'](Md['g_ba_c'](x_ab_1))
    x_aba_2 = Md['g_ba_d2'](Md['g_ba_c'](x_ab_2))
    x_aba = x_aba_1 + x_aba_2

    x_bab_1 = Md['g_ab_d'](Md['g_ab_c1'](x_ba))
    x_bab_2 = Md['g_ab_d'](Md['g_ab_c2'](x_ba))
    x_bab = F.concat((x_bab_1, x_bab_2), axis=1)#3ch,3ch⇒6ch

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
        Md['g_ab_c1'].cleargrads()
        Md['g_ab_c2'].cleargrads()
        Md['g_ab_d'].cleargrads()
        Md['g_ba_c'].cleargrads()
        Md['g_ba_d1'].cleargrads()
        Md['g_ba_d2'].cleargrads()

        gen_loss.backward()
        Op['g_ab_c1'].update()
        Op['g_ab_c2'].update()
        Op['g_ab_d'].update()
        Op['g_ba_c'].update()
        Op['g_ba_d1'].update()
        Op['g_ba_d2'].update()
 

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

def compute_loss_entro_X12(x_1,x_2):

    delta = chainer.Variable(1e-6 * cp.ones((x_1.shape[0],x_1.shape[2],x_1.shape[3]), dtype = np.float32 ))

    #ch平均
    x_1n = F.relu(F.mean(x_1, axis=1))
    x_2n = F.relu(F.mean(x_2, axis=1))

    #ノーマライズ
    #x_1n = x_1m / (F.sum(x_1m) / x_1m.shape[0] + delta) * 1.0 
    #x_2n = x_2m / (F.sum(x_2m) / x_2m.shape[0] + delta) * 1.0 

    loss_1 = x_1n * F.log( (x_1n + x_2n + delta) / (x_1n + delta) ) 
    loss_2 = x_2n * F.log( (x_1n + x_2n + delta) / (x_2n + delta) ) 

    loss = F.mean(loss_1 + loss_2) 

    return loss

   
#＜テスト部分スクリプト＞            
def run_test():

    x_a = chainer.Variable(xp.asarray(DataA['x']))
    x_ax_1 = Md['g_ab_c1'](x_a)
    x_ax_2 = Md['g_ab_c2'](x_a)
    x_ab_1 = Md['g_ab_d'](x_ax_1)
    x_ab_2 = Md['g_ab_d'](x_ax_2)
    x_ab = F.concat((x_ab_1, x_ab_2), axis=1)#3ch,3ch⇒6ch

    #中間のエントロピー的なものを計算
    #entro_X12 = compute_loss_entro_X12(x_ax_1, x_ax_2)
    #entro_X12 = cp.asnumpy(entro_X12.data)

    #numpy化
    DataA['y'] = cp.asnumpy(x_ab.data)
    DataA['y1'] = cp.asnumpy(x_ab_1.data)
    DataA['y2'] = cp.asnumpy(x_ab_2.data)

    #DataA['m1'] = np.mean(cp.asnumpy(x_ax_1.data),axis=1, keepdims= True)*100
    #DataA['m2'] = np.mean(cp.asnumpy(x_ax_2.data),axis=1, keepdims= True)*100

    #ピーク位置検出
    DataA['posi1'] = get_posi_from_img(DataA['y1'], threshold = Gv.Peek_threshold )
    DataA['posi2'] = get_posi_from_img(DataA['y2'], threshold = Gv.Peek_threshold )

    #円重ね描き
    DataA['x_y_circle'] = add_circle_from_posi(DataA['x'], DataA['posi1'], color = Gv.Cir_color[0])
    DataA['x_y_circle'] = add_circle_from_posi(DataA['x_y_circle'], DataA['posi2'], color = Gv.Cir_color[1])

    #bokeh描画
    imgs = [DataA['x_y_circle'][0], DataA['y1'][0], DataA['y2'][0]]
    infos = ['iteration = ' + str(Iteration), 'countA = ' + str(len(DataA['posi1'])), 'countB = ' + str(len(DataA['posi2']))]
    bokeh_update_img(imgs = imgs , infos = infos)
    





