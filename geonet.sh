#!/bin/bash
set -e

# Change directory to where you want the data
cd data/

# Function to download from Google Drive using curl
gdrive_download () {
    FILEID=$1
    FILENAME=$2
    echo "Downloading ${FILENAME}..."
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILEID}" > /dev/null
    CONFIRM=$(awk '/_warning_/ {print $NF}' ./cookie)
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" -o "${FILENAME}"
    rm cookie
}

# Download data1.zip
gdrive_download "1lWT6jZMsdZ3kOjbE1vQteA94ppIqhzy_" "data1.zip"

# Download data2.zip
gdrive_download "1G2uPo5ebS8gIhRIw2Gbw-QmI3QhP8ZDW" "data2.zip"

# Unzip the downloaded files
echo "Unzipping..."
unzip data1.zip
unzip data2.zip

# Organize the files
echo "Organizing files..."
mkdir -p nyu_geonet
mv data1/* nyu_geonet/
mv data2/* nyu_geonet/
rmdir data1 data2

# Cleanup
echo "Cleaning up..."
rm data1.zip data2.zip

echo "Done! NYU GeoNet dataset is ready under data/nyu_geonet/"
