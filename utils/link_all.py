import os
from os.path import join


def link_folder(args):
    "软链接文件夹(链接下面的mp4文件)"
    os.makedirs(args.dest, exist_ok=True)
    for home, folders, files in os.walk(args.source):
        # 创建对应文件夹
        for folder in folders:
            new_file_path = join(home, folder).replace(args.source, args.dest)
            os.makedirs(new_file_path, exist_ok=True)

        for file in files:
            old_file_path = join(home, file)
            new_file_path = old_file_path.replace(args.source, args.dest)
            cmd = f"ln -s {old_file_path} {new_file_path}"
            os.system(cmd)


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('--source', type=str, required=True)
    parse.add_argument('--dest', type=str, required=True)
    args = parse.parse_args()

    link_folder(args)
