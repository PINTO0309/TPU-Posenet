#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1feTGCIBtXkLXirnwvZU6oyMUFbLxS3nO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1feTGCIBtXkLXirnwvZU6oyMUFbLxS3nO" -o posenet_mobilenetv1.zip
unzip posenet_mobilenetv1.zip
rm posenet_mobilenetv1.zip
