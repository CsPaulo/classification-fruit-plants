#!/bin/bash
echo "Instalando Python 3.8..."
apt-get update
apt-get install -y python3.8 python3.8-distutils python3.8-venv
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
