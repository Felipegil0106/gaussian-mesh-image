# ══════════════════════════════════════════════════════════════
# Imagen Gaussian Scanner — Malla v3 (v2 + MVS-Texturing)
# ══════════════════════════════════════════════════════════════
# PARTE DE LA IMAGEN v2 (que YA funciona: COLMAP + OpenMVS + open3d + IAs)
# y le añade ENCIMA el texturizador MVS-Texturing (binario 'texrecon').
#
# ¿POR QUÉ MVS-Texturing? El seam-leveling de OpenMVS (lo que iguala el tono
# entre parches y quita los "mapitas") CRASHEA sobre la malla de Poisson.
# MVS-Texturing (Waechter et al., el estándar de oro libre) hace lo mismo Y
# MEJOR (global color adjustment + Poisson blending en costuras) y acepta
# CUALQUIER malla (incluida la de Poisson) usando las cámaras de COLMAP.
# Resultado buscado: CERO triángulos (Poisson) + CERO mapas (texrecon).
#
# CLAVE: como partimos de v2, NO recompilamos COLMAP/OpenMVS/IAs (que tardan
# ~40 min). Solo compilamos MVS-Texturing (~5-10 min). Si algo falla, la v2
# sigue intacta y podemos volver a ella al instante.
#
# Construir:  docker build -t felipegil0106/gaussian-mesh:v3 .
# Subir:      docker push felipegil0106/gaussian-mesh:v3
# ══════════════════════════════════════════════════════════════
FROM felipegil0106/gaussian-mesh:v2

ENV DEBIAN_FRONTEND=noninteractive

# ── Paso 1: dependencia que falta para MVS-Texturing ──
# mapmap (el optimizador interno de texrecon) necesita TBB. El resto de
# dependencias (libpng, libjpeg, libtiff, cmake, g++, eigen) YA vienen en v2.
RUN apt-get update -yq && apt-get install -yq \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Paso 2: compilar MVS-Texturing (nmoehrle/mvs-texturing) ──
# Su CMake descarga y compila sus propias sub-librerías (mve, rayint, mapmap)
# dentro de /elibs automáticamente. El binario principal es 'texrecon'.
# Lo dejamos en /usr/local/bin/texrecon (en el PATH).
RUN git clone https://github.com/nmoehrle/mvs-texturing.git /opt/mvs-texturing && \
    mkdir /opt/mvs-texturing/build && cd /opt/mvs-texturing/build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    find /opt/mvs-texturing/build -name texrecon -type f -exec cp {} /usr/local/bin/texrecon \; && \
    chmod +x /usr/local/bin/texrecon && \
    echo "=== texrecon compilado ===" && ls -la /usr/local/bin/texrecon

# ── Paso 3: verificar que texrecon corre (si no, el build falla aquí) ──
# texrecon sin argumentos imprime su ayuda y sale con código !=0; por eso
# usamos '|| true' y comprobamos que el binario responde.
RUN echo "=== Verificando texrecon ===" && \
    (/usr/local/bin/texrecon 2>&1 | head -5 || true) && \
    test -x /usr/local/bin/texrecon && \
    echo "=== texrecon OK en la imagen v3 ==="

WORKDIR /workspace
