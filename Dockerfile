# ══════════════════════════════════════════════════════════════
# Imagen Gaussian Scanner — Malla v2 (COLMAP + OpenMVS + IAs, todo adentro)
# ══════════════════════════════════════════════════════════════
# Igual que la v1 (COLMAP + OpenMVS precompilados) PERO ahora también trae
# las IAs de post-proceso (LaMa + Real-ESRGAN) ya instaladas y con sus modelos
# descargados. Así el worker NO instala nada en cada arranque → cero errores
# de instalación y renders más rápidos.
#
# Construir:  docker build -t TU_USUARIO/gaussian-mesh:v2 .
# Subir:      docker push TU_USUARIO/gaussian-mesh:v2
# ══════════════════════════════════════════════════════════════
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── Paso 1: sistema base + COLMAP + dependencias de OpenMVS ──
# (COLMAP de apt = el que ya usamos y sirve para la parte sparse)
# libgl1/libgomp1/libusb = las necesita open3d para importar en un servidor
#   sin pantalla (si faltan, open3d falla con 'libGL.so not found').
RUN apt-get update -yq && apt-get install -yq \
    build-essential git cmake wget ffmpeg pkg-config \
    colmap xvfb \
    libpng-dev libjpeg-dev libtiff-dev \
    libglu1-mesa-dev libglew-dev libglfw3-dev \
    libboost-iostreams-dev libboost-program-options-dev \
    libboost-system-dev libboost-serialization-dev libboost-thread-dev \
    libopencv-dev \
    libgmp-dev libmpfr-dev zlib1g-dev \
    python3-dev \
    libgl1 libgomp1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Paso 2: Eigen 3.4 (versión EXACTA que pide OpenMVS) ──
RUN git clone https://gitlab.com/libeigen/eigen --branch 3.4 /tmp/eigen && \
    mkdir /tmp/eigen_build && cd /tmp/eigen_build && \
    cmake . /tmp/eigen -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ && \
    make && make install && \
    cd / && rm -rf /tmp/eigen_build /tmp/eigen

# ── Paso 3: CGAL v6.0.1 (versión EXACTA de la receta oficial) ──
RUN git clone https://github.com/cgal/cgal --branch=v6.0.1 /tmp/cgal && \
    mkdir /tmp/cgal_build && cd /tmp/cgal_build && \
    cmake . /tmp/cgal && \
    make && make install && \
    cd / && rm -rf /tmp/cgal_build /tmp/cgal

# ── Paso 4: VCGLib (no se compila, solo se clona y se referencia) ──
RUN git clone https://github.com/cdcseacave/VCG.git /opt/vcglib

# ── Paso 4b: nanoflann (dependencia nueva de OpenMVS master) ──
# Es header-only, compila en segundos. Instala el nanoflannConfig.cmake
# que OpenMVS busca con FIND_PACKAGE(nanoflann REQUIRED).
RUN git clone https://github.com/jlblancoc/nanoflann.git --branch v1.5.5 /tmp/nanoflann && \
    mkdir /tmp/nanoflann_build && cd /tmp/nanoflann_build && \
    cmake . /tmp/nanoflann \
        -DNANOFLANN_BUILD_EXAMPLES=OFF \
        -DNANOFLANN_BUILD_TESTS=OFF && \
    make install && \
    cd / && rm -rf /tmp/nanoflann_build /tmp/nanoflann

# ── Paso 5: OpenMVS (compilar con CUDA, rama master estable) ──
RUN git clone https://github.com/cdcseacave/openMVS.git --branch master /tmp/openMVS && \
    sed -i 's/pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET ${MODULE_NAME})/pkg_check_modules(${PREFIX} IMPORTED_TARGET ${MODULE_NAME})/' /tmp/openMVS/libs/IO/CMakeLists.txt && \
    sed -i 's/cv::IMWRITE_JPEGXL_QUALITY/cv::IMWRITE_JPEG_QUALITY/g' /tmp/openMVS/libs/Common/Types.inl && \
    mkdir /tmp/openMVS_build && cd /tmp/openMVS_build && \
    cmake . /tmp/openMVS \
        -DCMAKE_BUILD_TYPE=Release \
        -DVCG_ROOT=/opt/vcglib \
        -DOpenMVS_USE_CUDA=ON \
        -DOpenMVS_BUILD_VIEWER=OFF \
        -DOpenMVS_USE_BREAKPAD=OFF \
        -DOpenMVS_ENABLE_TESTS=OFF \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
        -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/openMVS_build /tmp/openMVS

# ── Paso 6: librerías Python que el worker necesita ──
# (para descargar/subir, procesar imágenes y convertir la malla a .glb)
# Estas son ESENCIALES: si fallan, el build debe fallar (las necesitamos sí o sí).
RUN pip install --no-cache-dir \
    boto3 requests tqdm pillow "numpy<2" \
    opencv-python-headless trimesh

# ── Paso 6b: open3d (reconstrucción Poisson, anti-triángulos) ── OBLIGATORIO ──
# CAUSA EXACTA del fallo (vista en el log): open3d arrastra 'dash', que pide una
# versión nueva de 'blinker'. La imagen base trae blinker 1.4 instalado con un
# método antiguo (distutils) que pip NO puede desinstalar → "Cannot uninstall
# blinker 1.4" → exit 1. SOLUCIÓN: --ignore-installed blinker, para que pip NO
# intente borrar el viejo, solo instale el nuevo encima (no toca el del sistema).
# Sigue OBLIGATORIO: si falla, el build falla y vemos el error en el log.
RUN echo "=== ESPACIO ANTES ===" && df -h / && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker "open3d==0.19.0" "numpy<2" && \
    rm -rf /root/.cache/pip /tmp/* && \
    python3 -c "import open3d, numpy; print('open3d', open3d.__version__, '+ numpy', numpy.__version__, 'OK')"

# Los binarios de OpenMVS quedan en /usr/local/bin/OpenMVS
ENV PATH=/usr/local/bin/OpenMVS:$PATH

# ══════════════════════════════════════════════════════════════
# ── Paso 7: IAs de post-proceso (LaMa + Real-ESRGAN) ──
# Se instalan UNA SOLA VEZ aquí, al construir la imagen, en vez de en cada
# arranque del pod. Esto elimina de raíz los errores de instalación (red,
# scipy, etc.) y hace los renders más rápidos.
# La base ya trae torch 2.1.1 + torchvision + CUDA 12.1, así que NO se toca torch.
# ══════════════════════════════════════════════════════════════

# 7a. LaMa (relleno natural). --no-deps para que NO intente compilar el Pillow
#     viejo que pide (usa el torch/pillow/numpy de la base). + fire (su única
#     dependencia liviana extra).
RUN pip install --no-cache-dir --no-deps simple-lama-inpainting && \
    pip install --no-cache-dir fire

# 7b. Real-ESRGAN (nitidez) + basicsr, --no-deps para no tocar torch.
RUN pip install --no-cache-dir --no-deps basicsr realesrgan

# 7c. Dependencias livianas que basicsr/realesrgan necesitan (sin torch).
#     Todas de una vez para que no falte ninguna en tiempo de ejecución.
RUN pip install --no-cache-dir --no-deps \
    addict future lmdb pyyaml yapf scipy tqdm requests \
    tensorboard einops opencv-python gfpgan facexlib

# 7d. Parche permanente del bug de basicsr con torchvision nuevo:
#     basicsr importa 'torchvision.transforms.functional_tensor' (ya removido).
#     Creamos un shim físico en el paquete para que el import siempre funcione,
#     sin depender de un parche en tiempo de ejecución.
RUN python3 -c "import torchvision.transforms.functional as F, os, torchvision; \
p=os.path.join(os.path.dirname(torchvision.__file__),'transforms','functional_tensor.py'); \
open(p,'w').write('from torchvision.transforms.functional import rgb_to_grayscale\n'); \
print('shim functional_tensor creado en',p)"

# 7e. Pre-descargar los modelos de IA para que NO se bajen en cada arranque.
#     LaMa: big-lama.pt → ~/.cache/torch/hub/checkpoints/
#     Real-ESRGAN: RealESRGAN_x4plus.pth → carpeta 'weights' del paquete
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    wget -q -O /root/.cache/torch/hub/checkpoints/big-lama.pt \
      "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt" && \
    python3 -c "import os,realesrgan; d=os.path.join(os.path.dirname(realesrgan.__file__),'weights'); os.makedirs(d,exist_ok=True); print('weights dir:',d)" && \
    REALESRGAN_DIR=$(python3 -c "import os,realesrgan; print(os.path.join(os.path.dirname(realesrgan.__file__),'weights'))") && \
    wget -q -O "$REALESRGAN_DIR/RealESRGAN_x4plus.pth" \
      "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# 7f. Verificar que las IAs importan correctamente DENTRO de la imagen
#     (si las IAs fallan, el build falla aquí y nos enteramos antes de usarla).
RUN python3 -c "import simple_lama_inpainting; from realesrgan import RealESRGANer; from basicsr.archs.rrdbnet_arch import RRDBNet; import scipy, fire; print('=== IAs OK dentro de la imagen ===')"

# 7g. Verificar open3d (OBLIGATORIO): si no importa aquí, el build falla y nos
#     enteramos en el log, en vez de descubrirlo en el pod como hasta ahora.
RUN python3 -c "import open3d; print('=== open3d OK en la imagen:', open3d.__version__, '===')"

# Verificación: que los ejecutables existan (no debe hacer fallar el build)
RUN echo "=== Verificando OpenMVS ===" && \
    ls -la /usr/local/bin/OpenMVS/ && \
    echo "=== OpenMVS instalado OK ==="

WORKDIR /workspace
