#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1feTGCIBtXkLXirnwvZU6oyMUFbLxS3nO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1feTGCIBtXkLXirnwvZU6oyMUFbLxS3nO" -o posenet_mobilenetv1.zip
unzip posenet_mobilenetv1.zip
rm posenet_mobilenetv1.zip

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1mdUKcwFTckmoStQpS4SUihGaz7eUt-Xt" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1mdUKcwFTckmoStQpS4SUihGaz7eUt-Xt" -o deeplabv3.zip
unzip deeplabv3.zip
rm deeplabv3.zip

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1ZYoP824ZBNgpnX-K2LcE7XgjIOBYdcdk" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1ZYoP824ZBNgpnX-K2LcE7XgjIOBYdcdk" -o mobilenet_ssd_v2_coco.zip
unzip mobilenet_ssd_v2_coco.zip
rm mobilenet_ssd_v2_coco.zip

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1lAbSVsmG9ticeIicHqgDrBFru_pJeyzZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1lAbSVsmG9ticeIicHqgDrBFru_pJeyzZ" -o colorpalette.png

