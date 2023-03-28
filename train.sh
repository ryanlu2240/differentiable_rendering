#!/bin/bash

prompt="a newspaper with title at the top"
bg_img="img/newspaper_empty_bg.jpg"
src_img="img/newspaper_title.jpg"
src_imgs_folder="img/2"
workspace="trail/exp14"
python3 main.py --bg_img $bg_img --src_img $src_img --prompt "$prompt" --workspace $workspace --src_imgs_folder $src_imgs_folder --seed 10000 --lr 0.1