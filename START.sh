#!/bin/bash
cd ~/plant-disease-detector 2>/dev/null || cd /teamspace/studios/this_studio/plant-disease-detector
pip install -q -r requirements.txt
echo "=== Dataset (carpetas train/) ==="
find /teamspace /home/zeus -maxdepth 8 -type d -name train 2>/dev/null | head -5
