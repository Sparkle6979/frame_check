import os,glob
from xml.dom import minidom

def skip_check(imgpath,respath,fps):
    writepath = os.path.join(respath,'frame_check.txt')
    f = open(writepath,"w")

    alpic = os.listdir(imgpath)
    igs = sorted([os.path.join(imgpath,p) for p in alpic])
    # fmslist = [c+'/frames.info' for c in igs]
    fmslist = igs
    staffs = [minidom.parse(f) for f in fmslist]
    staffs = [s.getElementsByTagName("frame") for s in staffs]

    finmes = []
    for s in staffs:
        frames = sorted([float(f.getAttribute("pkt_pts_time")) for f in s])
        finmes.append(frames)
    
    for i,cam in enumerate(finmes):
        # print('camera: %s, 共计 %s 帧'%(i,len(cam)))
        f.write('camera: %s, 共计 %s 帧\n'%(i,len(cam)))
        for ind in range(0,len(cam)-1):
            if (cam[ind+1] - cam[ind]) - (1.0/fps) >= 1e-4:
                # print('第 %s 帧，该帧与下一帧间隔时间为 %s'%(ind+1,(cam[ind+1] - cam[ind])))
                f.write('第 %s 帧，该帧与下一帧间隔时间为 %s\n'%(ind+1,(cam[ind+1] - cam[ind])))
    
     
        f.write('-----------------------------------------------------------------------\n')

    f.close()

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--src", type=str, help='project_path',required=True)
    parse.add_argument("--dst", type=str, help='res_path',required=True)
    parse.add_argument("--fps", type=float, help='fps',required=True)

    args = parse.parse_args()
    
    # frame_info_gereator(args.src)
    skip_check(args.src,args.dst,args.fps)
