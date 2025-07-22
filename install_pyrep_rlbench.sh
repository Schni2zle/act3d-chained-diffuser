#!/bin/bash

# Install PyRep
cd PyRep
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
echo "export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT" >> ~/.bashrc
echo "export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc
source ~/.bashrc
pip install -r requirements.txt
pip install -e .
cd ..

# Install RLBench
cd RLBench
pip install -r requirements.txt
pip install -e .
cd ..

# Install dependencies
sudo apt-get update
sudo apt-get install -y xorg libxcb-randr0-dev libxrender-dev libxkbcommon-dev libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev

# Reboot to apply changes
echo "Rebooting system now..."
sudo reboot

