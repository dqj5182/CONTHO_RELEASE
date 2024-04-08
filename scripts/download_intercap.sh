#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# username and password input
echo -e "\nYou need to register at https://intercap.is.tue.mpg.de/"
read -p "Username:" username
read -p "Password:" password
read -p "Save directory:" save_dir
username=$(urle $username)
password=$(urle $password)
save_dir=$save_dir

mkdir $save_dir

# Download "RGBD_Images.zip"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=intercap&resume=1&sfile=RGBD_Images.zip' -O $save_dir'/RGBD_Images.zip' -P save_dir --no-check-certificate --continue 

# Download "Res.zip"
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=intercap&resume=1&sfile=Res.zip' -O $save_dir'/Res.zip' -P save_dir --no-check-certificate --continue 