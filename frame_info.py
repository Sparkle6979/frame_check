import os,glob


def frame_info_gereator(path):

    pt = os.path.join(path,'videos')
    infoname = sorted([v.replace(".mp4","") for v in os.listdir(pt)])
    # print(infoname)
    os.makedirs(path+'/infos',exist_ok = True)

    igt = os.path.join(path,'infos')

    videos = sorted(glob.glob(pt+'/*.mp4'))

    
    for (v,n) in zip(videos,infoname): 
        cmd = f"ffprobe -show_frames -select_streams v -show_entries frame=pkt_pts_time -of xml {v} >{igt}/{n}.info -loglevel quiet"
        print(f"生成 {n}.info 中...")
        os.system(cmd)

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--src", type=str, help='project_path',required=True)

    args = parse.parse_args()
    
    frame_info_gereator(args.src)

