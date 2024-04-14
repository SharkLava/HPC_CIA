#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <image_folder> <binary>" >&2
    exit 1
fi

image_folder="$1"
binary="$2"

for image_file in "$image_folder"/*.png; do
    # Check if the file exists
    if [ -f "$image_file" ]; then
        "$binary" "$image_file"
    fi
done
