# Whether to enable MKL, GPU, or TensorRT
# When TensorRT is enabled, GPU must be ON

# INFER_NAME=infer_v3
# INFER_NAME=eval_v3
INFER_NAME=infer_r2plus1d
if [ "$(uname)" == "Darwin" ]; then
    WITH_MKL=OFF
    WITH_GPU=OFF
    USE_TENSORRT=OFF

    LIB_DIR=/Users/xueyang02/Code/paddle-capi
    OPENCV_DIR=/Users/xueyang02/Code/opencv-src/build_opencv
    CUDA_LIB_DIR=YOUR_CUDA_LIB_DIR
    CUDNN_LIB_DIR=YOUR_CUDNN_LIB_DIR
    MODEL_DIR=/Users/xueyang02/Code/repo/xiaodu-hi/jetson/baseline_r2plus1d
    GRPC_INSTALL_DIR=/Users/xueyang02/Code/grpc_install
elif [ "$(uname)" == "Linux" ]; then
    WITH_MKL=OFF
    WITH_GPU=ON
    USE_TENSORRT=OFF

    LIB_DIR=/mnt/xueyang/Code/paddle_inference
    OPENCV_DIR=/mnt/xueyang/Code/opencv-4.2.0/build
    CUDA_LIB_DIR=/usr/local/cuda/lib64
    CUDNN_LIB_DIR=/usr/lib64
    MODEL_DIR=/mnt/xueyang/Code/xiaodu-hi/jetson/baseline_r2plus1d
    GRPC_INSTALL_DIR=/mnt/xueyang/Code/grpc_install
fi


sh run_impl.sh ${INFER_NAME} ${LIB_DIR} ${OPENCV_DIR} ${MODEL_DIR} ${WITH_MKL} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${USE_TENSORRT} ${GRPC_INSTALL_DIR}
