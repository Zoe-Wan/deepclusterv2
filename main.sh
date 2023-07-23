# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
BASE="/mnt/lustre/suxiu/wzy/drive/"


DIR="/mnt/lustre/suxiu/wzy/self-cifar/data"
LR=0.01
WD=-5
K=100
ALPHA=0.7
BATCH_SIZE=32
WORKERS=1
EXP="exp_sp_3/"
CKPT=${EXP}"checkpoint.pth.tar"

PYTHON="python3"
JOB=${BASE}"main.py"

mkdir -p ${EXP}

/mnt/lustre/suxiu/wzy/self-cifar/script/test.sh sp_3 1 1 "${PYTHON} ${JOB} ${DIR} --exp ${EXP}  \
  --lr ${LR} --wd ${WD} --batch ${BATCH_SIZE} --k ${K} --sobel --verbose --workers ${WORKERS} --alpha ${ALPHA}"

