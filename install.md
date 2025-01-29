# Install

```bash
mkdir -p thirdparty
git submodule add -f https://github.com/cvg/Hierarchical-Localization.git thirdparty/Hierarchical-Localization
git submodule add -f https://github.com/colmap/colmap.git thirdparty/colmap
git submodule update --init --recursive
# vim cmakelist add set(CMAKE_CUDA_ARCHITECTURES 89)
# ceres-solver 2.2.0 build from source

# GLIBCXX_3.4.30
conda install -c conda-forge libstdcxx-ng=12
# colmap
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DMVS_ENABLED=OFF -GNinja
ninja
sudo ninja install
Downgrade pycolmap to 3.11.1
# fix
gh pr checkout 446
cd Hierarchical-Localization/
python -m pip install -e .
pycolmap to 3.10.0
pip install PyMCubes==0.1.0
pip install skimage

```
