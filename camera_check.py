import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

import cv2
import os,csv,glob,json
from utils.img_transf import match_image
from model.dc_net import DCN_Net
import multiprocessing
from xml.dom import minidom

class bs_set():
    def __init__(self,s) -> None:
        self.ccnt = 0
        self.ids = s
        self.f = [i for i in range(0,s)]
    def find_f(self,ind):
        if ind == self.f[ind]:
            return ind
        else:
            self.f[ind] = self.find_f(self.f[ind])
            return self.f[ind]
    def union(self,*args):
        for i in range(0,len(args)-1):
            fxi = self.find_f(args[i])
            fyi = self.find_f(args[i+1])
            if fxi != fyi:
                self.f[fxi] = args[i+1]
                self.ccnt += 1

    def jud_al(self):
        return True if self.ccnt == self.ids - 1 else False
        
    def get_dict(self):
        dit = dict()
        for i in range(0,self.ids):
            fi = self.find_f(i)
            if fi not in dit:
                dit[fi] = []
            dit[fi].append(i)
        return dit



def orimg2num(img_path,model,tar,ignore_flag = True,gray_flag = True,rsz_flag = True):
    transfimg = match_image(img_path,ignore_flag,gray_flag,rsz_flag)
    
    # 如果转换失败，不进行预测
    if transfimg is None:
        return -1
    else:
        
        model.eval()
        transfimg = cv2.cvtColor(transfimg, cv2.COLOR_GRAY2RGB)
        h,w = transfimg.shape[:2]
        transfimg = cv2.resize(transfimg[:,int(w/2):int(3*w/4),:],(224,224))
        transfimg = transforms.ToPILImage()(transfimg)
        transfimg = transforms.ToTensor()(transfimg)
        transfimg = transfimg.unsqueeze(0)
        res = model(transfimg)

        # predicted：预测结果
        _,predicted = torch.max(res,axis=1)
        # val：置信度
        val,_ = torch.max(nn.Softmax(dim=1)(res),axis=1)
        
        return predicted[0].item() if val[0].item() >= tar else -1



def Solve(args):

    rpath = args.src
    dst = args.dst
    if not dst:
        dst = rpath
    dst = os.path.join(dst,'result')
    os.makedirs(dst,exist_ok=True)
    
    args.src = os.path.join(args.src,'images')
    print('文件路径: %s'%(args.src))
    camera_bag = [f for f in os.listdir(args.src) if not f.startswith('.')]

    image_path = sorted([os.path.join(args.src,bag) for bag in camera_bag])

    all_img = [sorted(glob.glob(img+'/*.jpg')) for img in image_path]  
    frameinfo = os.path.join(rpath,'infos')
    
    all_info = sorted(glob.glob(frameinfo+'/*.info'))
    
    process_num = args.imgnum if args.imgnum != -1 else len(all_img[0])

    minl,maxl = 1e9,-1
    
    # 进行了图文件的重命名
    allstaffs = []
    for (i,img_lst) in enumerate(all_img):
        nowframeinfo = all_info[i]
        if not args.noname:
            doc = minidom.parse(nowframeinfo)
            staffs = doc.getElementsByTagName("frame")
            staffs = [st.getAttribute("pkt_pts_time") for st in staffs]
            staffs = sorted([(str(t).replace(':','').replace('.',''))[:-3].zfill(8) for t in staffs])
        else:
            staffs = img_lst
        allstaffs.append(staffs)

        if args.rename:
            for (j,img) in enumerate(img_lst):
                if j < len(staffs):
                    # 进行图片的重命名
                    os.rename(img,image_path[i]+'/'+staffs[j]+'.jpg')
                    


    all_img = [sorted(glob.glob(img+'/*.jpg')) for img in image_path]
        

        
    print(len(all_img))

    for (i,img_lst) in enumerate(all_img):
        print('%s文件夹图片数量为%s'%(str(i).zfill(3),str(len(img_lst)).zfill(5)),end='\t')
        if len(img_lst) > maxl: maxl = len(img_lst)
        if len(img_lst) < minl: minl = len(img_lst)
        if (i+1)%4 == 0:
            print('')
    process_num = minl
    # if (minl != maxl and minl < process_num):
    #     print('\n文件夹图片数量存在问题，给定处理值为%s，最小值为%s'%(process_num,minl))
    #     sys.exit(0)

    
    
    print('\n模型加载路径: %s'%(args.model))
    print('模型加载ing....')
    model = DCN_Net(output_size=10)
    model.load_state_dict(torch.load(args.model))
    print('模型加载完毕')

    
    pool = multiprocessing.Pool(args.pool)
    finr = []


    print('图片处理中，线程数为：%s'%(args.pool))
    # print('单一文件夹图片数量：%s'%(len(all_img[0])))
    print('相机数量：%s'%(len(all_img)))
    
    from tqdm import tqdm
    for (i,imgfile) in enumerate(all_img):
        
        process_file = imgfile[(-1)*process_num::args.step] if args.reverse else imgfile[:process_num:args.step]          
        processes = [pool.apply_async(orimg2num,args=(imgpath,model,args.bench,True,True,True,)) for imgpath in process_file]
        
        aftimg = dict()
        print('文件夹%s:'%(str(i).zfill(3)))

        cnt = 0
        for p in tqdm(processes):
            aftimg[allstaffs[i][cnt]] = p.get()
            cnt = cnt+args.step
        finr.append(aftimg)


    print('图片处理完毕！')


    allfps = set()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    for obag in finr:
        for key in obag.keys():
            allfps.add(key)

    allfps = list(allfps)
    allfps.sort()

    newfinr = []
    for obag in finr:
        tmpl = []
        for val in allfps:
            if val not in obag:
                obag[val] = -1
            tmpl.append(obag[val])
        newfinr.append(tmpl)
                
    judset = bs_set(len(newfinr))

    
    npres = np.array(newfinr).T

    checktxt = open(os.path.join(dst,'check.txt'),'w')
    
    with open(os.path.join(dst,'result.csv'),'w') as f:
        writer = csv.writer(f)
        writer.writerows(npres)


    for i in range(0,len(npres)):
        frame = len(all_img[0])-process_num+i if args.reverse else i
        nowl,indxs = npres[i],[]

        for j in range(0,len(nowl)):
            if nowl[j] != -1:
                indxs.append(j)
        
        for k in range(0,len(indxs)):
            for z in range(k+1,len(indxs)):
                if (nowl[indxs[k]] == nowl[indxs[z]]) and (judset.find_f(indxs[k]) != judset.find_f(indxs[z])):
                    judset.union(indxs[k],indxs[z])
                    print('第 %s\t ms，%s 与 %s 同步'%(allfps[frame],indxs[k],indxs[z]))
                    checktxt.write('第 %s\t ms，%s 与 %s 同步\n'%(allfps[frame],indxs[k],indxs[z]))
                elif nowl[indxs[k]] != nowl[indxs[z]] and (1 < abs(nowl[indxs[k]] - nowl[indxs[z]]) < 9):
                    print('第 %s\t ms，%s 与 %s 相差 %s'%(allfps[frame],indxs[k],indxs[z],abs(nowl[indxs[k]]-nowl[indxs[z]])))
                    checktxt.write('第 %s\t ms，%s 与 %s 相差 %s\n'%(allfps[frame],indxs[k],indxs[z],abs(nowl[indxs[k]]-nowl[indxs[z]])))

        if judset.jud_al():
            print('全部同步')
            checktxt.write('全部同步\n')
            break
    checktxt.close()
    
    finset = judset.get_dict()
    
    for (key,val) in finset.items():
        finset[key] = [image_path[ind] for ind in val]
        print('同步集合: ',finset[key])
    
    with open(os.path.join(dst,'finset.json'),'w',encoding='utf-8') as f:
        f.write(json.dumps(finset,indent=4,ensure_ascii=False))

    print('同步结果写至 %s 文件夹下finset.json 文件中'%(dst))
    

if __name__ == '__main__':

    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--src", type=str, help='ori_image_path',required=True)
    parse.add_argument("--dst", type=str, help='res_save_path',required=False)
    parse.add_argument("--imgnum", type=int, help='image_number',required=False,default=-1)
    parse.add_argument("--step",type=int, help='step_size',required=False,default=1)
    parse.add_argument("--pool", type=int, help='pool_number',required=True,default=3)
    parse.add_argument("--model", type=str, help='pretrain_model_path',required=True)
    parse.add_argument("--bench", type=float, help='benchmark',required=False,default=0.9)
    parse.add_argument("--reverse",action='store_true',default=False)
    parse.add_argument("--noname",action='store_true',default=False)
    parse.add_argument("--rename",action='store_true',default=False)

    args = parse.parse_args()

    Solve(args)
