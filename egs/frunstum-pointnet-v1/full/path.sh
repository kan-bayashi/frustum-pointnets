export PRJ_ROOT="${PWD}/../../.."
source "${PRJ_ROOT}"/venv/bin/activate
export PYTHONPATH=$PRJ_ROOT/models:$PRJ_ROOT/kitti:$PRJ_ROOT/train:$PRJ_ROOT/mayavi
export PATH=$PRJ_ROOT/train:$PRJ_ROOT/kitti:$PRJ_ROOT/utils:$PRJ_ROOT/train/kitti_eval:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_HOME=/usr/local/cuda
