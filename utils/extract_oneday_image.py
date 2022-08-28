import os
from multiprocessing import Pool
from os.path import join

"""
将采集的一天的数据链接到另外的文件夹下面并拆成图片
--source /nas/dataset/xiaomi_test/20220518
--dest /nas/dataset/detectSyn_0518
"""
def link_folder(args):
    "软链接文件夹(链接下面的mp4文件)"
    os.makedirs(args.dest, exist_ok=True)
    for home, folders, files in os.walk(args.source):
        # 创建对应文件夹
        for folder in folders:
            if not folder == "labels" and not folder == "@eaDir":
                new_file_path = join(home, folder).replace(args.source, args.dest)
                os.makedirs(new_file_path, exist_ok=True)
        for file in files:
            old_file_path = join(home, file)
            if not "labels" in old_file_path:
                new_file_path = old_file_path.replace(args.source, args.dest)
                cmd = f"ln -s {old_file_path} {new_file_path}"
                os.system(cmd)
def ffmpeg_video(args):
    video_path = args[0]
    output_path = args[1]
    frame_path = args[2]
    frame = args[3]
    os.makedirs(output_path, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -vsync passthrough -enc_time_base -1 -q:v 1 -start_number 0 {output_path}/%06d.jpg -loglevel quiet"
    print(f'processing {video_path}')
    os.system(cmd)
    if frame:
        cmd = f"ffprobe -show_frames -select_streams v -show_entries frame=pkt_pts_time -of xml {video_path} >{frame_path}.info -loglevel quiet"  
        os.system(cmd)
        
def extract_all_videos(args):
    if args.frame:
        os.makedirs(args.dest+'/infos',exist_ok = True)
    "Pool多线程处理视频"
    threads = []
    for home, folders, files in os.walk(args.dest):
        for file in files:
            if ".mp4" in file:
                old_file_path = join(home, file)
                new_file_path = old_file_path.replace("videos", "images").replace(
                    ".mp4", ""
                )
                frame_file_path = old_file_path.replace("videos","infos").replace(
                    ".mp4", ""
                )
                threads.append([old_file_path, new_file_path,frame_file_path,args.frame])
    with Pool(args.threads_num) as pool:
        pool.map(ffmpeg_video, threads)
        pool.close()
        pool.join()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--threads_num", type=int, default=8)
    parser.add_argument("--extract", type=bool, default=True)
    parser.add_argument("--frame", action='store_true',default=False)
    args = parser.parse_args()
    link_folder(args)
    if args.extract:
        extract_all_videos(args)
