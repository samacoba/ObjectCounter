#coding:utf-8
#実行用スクリプト
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable
from chainer import optimizers
import numpy as np
import data
import time
from PIL import Image





stop_flag = False

#モデル読み込み、初期設定
def run_set_model():

    gpu = 0 # 0：gpu使用、-1：gpu不使用

    global Md#モデル
    global xp
    global Op#オプティマイザ
    global iteration

    Md ={}
    Op ={}

    xp = cuda.cupy
    cuda.get_device(gpu).use()

    from model import Generator,Discriminator

    Md['g_ab'] = Generator()
    Md['g_ba'] = Generator()
    Md['d_a'] = Discriminator()
    Md['d_b'] = Discriminator()

    for key in Md.keys():
        Md[key].to_gpu()
        Op[key] = optimizers.Adam(2e-4, beta1=0.5, beta2=0.999)
        Op[key].setup(Md[key])
        Op[key].add_hook(chainer.optimizer.WeightDecay(1e-4))  


    if True:#ディーラ初期化
        w_init = np.ones((4,1),dtype=np.float32)#Wを全部1で初期化
        Md['de'] = L.Linear(1,4, nobias = True, initialW = w_init)
        Md['de'].to_gpu()

        Op['de'] = optimizers.Adam(1e-3, beta1=0.5, beta2=0.999)
        Op['de'].setup(Md['de'])

    Md['decon_core'] = get_decon_core()

    iteration = 0

#トレーニングとテスト
def run_train(nLoop = 50):

    global batchSize
    global N_train
    global loop
    global stop_flag
    global start_time

    batchSize = 100
    N_train = 2000   
    
    start_time = time.time()

    for loop in range(nLoop):

        if(stop_flag == True):#ストップフラグがTrueでトレーニング停止
            print('学習中断')
            break 
        
        #＜トレーニング＞
        
        for i in range(0, N_train, batchSize):            

            run_disco_update()

        run_test()

    print('学習終了')
    stop_flag = False

#トレーニングスレッド実行
def run_train_thread(target = run_train,**kwargs):
    global train_thread
    if ('train_thread' not in globals()) or (train_thread.is_alive() == False):
        #別スレッドでトレーニングを実行する        
        import threading
        train_thread = threading.Thread(target = target,kwargs = kwargs)
        train_thread.start()#トレーニングスタート
        print('学習開始')  


#DiscoGAN UPdate
def run_disco_update():

    batchSize = 100
    # read data       

    DataN = data.get_data_N_rand(DataA, N_pic = batchSize, imgH = 64, imgW = 64, keys =['x'])
    x_a = Variable(xp.asarray(DataN['x'])) 

    
    if True:#ディーラサンプリング
        x_b = get_dealer_sampling()
    else:
        DataN = data.get_data_N_rand(DataB, N_pic = batchSize, imgH = 64, imgW = 64, keys =['x'])        
        x_b = Variable(xp.asarray(DataN['x']))


    # conversion
    x_ab = Md['g_ab'](x_a)
    x_ba = Md['g_ba'](x_b)

    # reconversion
    x_aba = Md['g_ba'](x_ab)
    x_bab = Md['g_ab'](x_ba)

    # reconstruction loss
    recon_loss_a = F.mean_squared_error(x_a, x_aba)
    recon_loss_b = F.mean_squared_error(x_b, x_bab)

    # discriminate
    y_a_real, feats_a_real = Md['d_a'](x_a)
    y_a_fake, feats_a_fake = Md['d_a'](x_ba)

    y_b_real, feats_b_real = Md['d_b'](x_b)
    y_b_fake, feats_b_fake = Md['d_b'](x_ab)

    # GAN loss
    gan_loss_dis_a, gan_loss_gen_a ,gan_loss_del_a = compute_loss_gan(y_a_real, y_a_fake)
    feat_loss_a = compute_loss_feat(feats_a_real, feats_a_fake)

    gan_loss_dis_b, gan_loss_gen_b ,gan_loss_del_b = compute_loss_gan(y_b_real, y_b_fake)
    feat_loss_b = compute_loss_feat(feats_b_real, feats_b_fake)

    # compute loss
    global iteration
    if iteration < 10000:
        rate = 0.01
    else:
        rate = 0.5

    total_loss_gen_a = (1.-rate)*(0.1*gan_loss_gen_b + 0.9*feat_loss_b) + rate * recon_loss_a
    total_loss_gen_b = (1.-rate)*(0.1*gan_loss_gen_a + 0.9*feat_loss_a) + rate * recon_loss_b

    gen_loss = total_loss_gen_a + total_loss_gen_b
    dis_loss = gan_loss_dis_a + gan_loss_dis_b

    del_loss = gan_loss_del_a + gan_loss_del_b

    if iteration % 3 == 0:
        Md['d_a'].cleargrads()
        Md['d_b'].cleargrads()
        dis_loss.backward()
        Op['d_a'].update()
        Op['d_b'].update()
    else:
        Md['g_ab'].cleargrads()
        Md['g_ba'].cleargrads()
        gen_loss.backward()
        Op['g_ab'].update()
        Op['g_ba'].update()

        #ディーラアップデート
        if True:#ディーラサンプリング
            Md['de'].cleargrads()
            del_loss.backward()
            Op['de'].update()
   
    iteration = iteration + 1

def compute_loss_gan(y_real, y_fake):
    batchsize = y_real.shape[0]
    loss_dis = 0.5 * F.sum(F.softplus(-y_real) + F.softplus(y_fake))
    loss_gen = F.sum(F.softplus(-y_fake))
    loss_del = F.sum(F.softplus(y_real))
    return loss_dis / batchsize, loss_gen / batchsize, loss_del / batchsize

def compute_loss_feat(feats_real, feats_fake):
    losses = 0
    for feat_real, feat_fake in zip(feats_real, feats_fake):
        feat_real_mean = F.sum(feat_real, 0) / feat_real.shape[0]
        feat_fake_mean = F.sum(feat_fake, 0) / feat_fake.shape[0]
        l2 = (feat_real_mean - feat_fake_mean) ** 2
        loss = F.sum(l2) / l2.size
        losses += loss
    return losses


   
#＜テスト部分スクリプト＞            
def run_test():

    with chainer.using_config('train', False):
        x_batch = chainer.Variable(xp.asarray(DataA['x']))
        y_batch = Md['g_ab'](x_batch)
    
    y_batch.to_cpu()
    
    DataA['y'] = y_batch.data    
    
    #極大値の位置を抽出して、赤いサークルを元絵に重ねる
    DataA['y_point'] = data.get_local_max_point(F.mean(DataA['y'], axis = 1, keepdims = True).data, threshold = 0.2)
    DataA['y_circle'] = data.draw_circle(DataA['y_point'])
    DataA['x_y_circle'] = DataA['x'].copy()
    DataA['x_y_circle'][:,0,:,:]=DataA['x_y_circle'][:,0,:,:] + DataA['y_circle'][:,0,:,:]
    
    #bokehアップデート
    if "bokeh_view" in globals():
        prob = xp.asnumpy(F.softmax(Md['de'].W.data.transpose(1,0)).data.reshape(4))
        bokeh_view.update_img(imgs = [DataA['x_y_circle'][0], DataA['y'][0]], prob = prob,trainN = (loop+1)*N_train, count = DataA['y_point'].sum())





#ディーラサンプル生成
def get_dealer_sampling(N_pic=100, imgH=64, imgW=64, N_card = 4): 

    
    thres =[0.99995, 0.9999, 0.9998, 0.9995]#*512で13,26,52,131個相当

    #＜ランダム点画像の生成＞
    img_r = xp.random.rand(N_pic, imgW*imgH).astype(np.float32)#100枚分の0-1乱数作成
    img_p = xp.zeros((N_card, N_pic, imgW*imgH)).astype(np.float32)#4*100枚分のイメージメモリ確保

    for i, thre in enumerate(thres):#閾値よりも高いものだけ1を代入
        img_p[i][img_r >= thre] = 1

    #点画像変形　(N_card, N_pic, imgW*imgH,) ⇒ (N_pic, imgW*imgH, N_card)
    img_p = chainer.Variable(img_p.transpose((1,2,0)))
    
    #＜サンプリング係数の生成＞
    #100個の「１」を作成
    x_one = xp.ones((N_pic,1), dtype = np.float32)
    #「１」をディーラーを通したあとsoftmaxで0-1確率にする
    card_prob = F.softmax(Md['de'](x_one))
    #gumbel_softmaxを通してサンプリング
    card_gum = F.gumbel_softmax(F.log(card_prob),tau = 0.2)
    #サンプリング係数の画像化　(N_pic, N_card) ⇒ (N_pic, imgW*imgH, N_card)
    card_gum_b = F.broadcast_to(F.reshape(card_gum,(N_pic, 1, N_card)),img_p.shape)
    
    #＜ランダム点画像とサンプリング係数画像の合成＞
    #ランダム点画像とサンプリング係数をかけて、合成(sum)し、２次元画像へ変形
    img_p_sum = F.reshape(F.sum(img_p * card_gum_b, axis =2),(N_pic,1,imgH,imgW))

    #点⇒ガウス球へ変形
    img_core  = Md['decon_core'](img_p_sum)*255
    img_core = F.broadcast_to(img_core, (N_pic, 3, imgH, imgW))
    
    return img_core


#ディーラdeconコア生成用
def get_decon_core():
    import math
    sig=3.0
    max_xy = 15
    c_xy= 7
    sig2=sig*sig
    core=np.zeros((max_xy, max_xy),dtype = np.float32)
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1,1,core.shape[0],core.shape[1]))
    decon_core = L.Deconvolution2D(1, 1,ksize=max_xy, stride=1, pad=c_xy, initialW=core).to_gpu()
    
    return decon_core