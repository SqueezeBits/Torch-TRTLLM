URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/$(echo ${TRT_VER} | cut -d '.' -f 1-3)/tars/TensorRT-${TRT_VER}.Linux.x86_64-gnu.cuda-$(echo ${NV_CUDA_LIB_VERSION} | cut -d '.' -f 1-2).tar.gz"
echo "Downloading from ${URL} ..."
wget -c --no-verbose ${URL} -O /root/.cache/TensorRT.tar
echo "Unzipping TensorRT.tar ..."
tar -xf /root/.cache/TensorRT.tar -C /usr/local/
mv /usr/local/TensorRT-${TRT_VER} /usr/local/tensorrt
pip install /usr/local/tensorrt/python/tensorrt-*-cp${PYTHON_MAJOR_VERSION}${PYTHON_MINOR_VERSION}-*.whl
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/tensorrt/lib' >> "${ENV}"
