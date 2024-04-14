#!/bin/bash

for file in "[RH] Nichijou - 16 [A7D81A13].mkv"*.png; do
    new_file=$(echo "$file" | sed 's/\[RH\] Nichijou - 16 \[A7D81A13\]\.mkv//; s/\.png//')
    mv "$file" "$new_file.png"
done

echo "Files renamed successfully!"
