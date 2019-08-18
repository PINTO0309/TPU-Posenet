#!/bin/bash

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1YTIGtrywH9rgIz30N-PqO--ohMroTZgZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1YTIGtrywH9rgIz30N-PqO--ohMroTZgZ" -o test_medias.zip
unzip test_medias.zip
rm test_medias.zip
