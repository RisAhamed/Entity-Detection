# Entity-Detection

sudo apt-get update
sudo apt-get install -y libgl1


sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6


conda update libstdcxx-ng

conda install -c conda-forge libstdcxx-ng





LD_PRELOAD=/opt/conda/lib/libstdc++.so.6 python process_video.py

export LD_PRELOAD=/opt/conda/lib/libstdc++.so.6

export LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH



strings /opt/conda/lib/libstdc++.so.6 | grep GLIBCXX_3.4.32




