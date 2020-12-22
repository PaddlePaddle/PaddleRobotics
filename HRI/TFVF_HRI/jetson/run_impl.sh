mkdir -p build
cd build
rm -rf *

INFER_NAME=$1
LIB_DIR=$2
OPENCV_DIR=$3
MODEL_FILE_DIR=$4
WITH_MKL=$5
WITH_GPU=$6
CUDNN_LIB=$7
CUDA_LIB=$8
USE_TENSORRT=$9
GRPC_INSTALL_DIR=${10}

cmake -DCMAKE_PREFIX_PATH=${GRPC_INSTALL_DIR} \
        -DPADDLE_LIB=${LIB_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DWITH_MKL=${WITH_MKL} \
        -DINFER_NAME=${INFER_NAME} \
        -DWITH_GPU=${WITH_GPU} \
        -DWITH_STATIC_LIB=OFF \
        -DUSE_TENSORRT=${USE_TENSORRT} \
        -DCUDNN_LIB=${CUDNN_LIB} \
        -DCUDA_LIB=${CUDA_LIB} \
        ..

make -j

if [ ${WITH_GPU} = "ON" ]; then
    ./${INFER_NAME} -dirname ${MODEL_FILE_DIR} -gpu
else
    ./${INFER_NAME} -dirname ${MODEL_FILE_DIR}
fi
