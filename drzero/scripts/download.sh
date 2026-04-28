# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


save_path=/home/peterjin/debug_cache

python download.py --savepath $savepath

cat $save_path/part_* > e5_Flat.index
