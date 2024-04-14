#!/bin/bash

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not installed. Please install it and try again."
    exit 1
fi

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <input.mkv> <start_time> <end_time> <number_of_frames> <output_directory>"
    exit 1
fi

file_path="$1"
start_time="$2"
end_time="$3"
number_of_frames="$4"
output_directory="$5"

input_file=$(basename "$file_path")

mkdir -p "$output_directory"

time_to_seconds() {
    IFS=: read -r hour minute second <<< "$1"
    echo $(( 10#$hour*3600 + 10#$minute*60 + 10#$second ))
}

start_seconds=$(time_to_seconds "$start_time")
end_seconds=$(time_to_seconds "$end_time")
duration=$(( end_seconds - start_seconds ))
interval=$(echo "scale=6; $duration / $number_of_frames" | bc)

echo "Extracting frames from $input_file every $interval seconds."

for ((i=0; i<number_of_frames; i++)); do
    current_offset=$(echo "scale=6; $start_seconds + $i * $interval" | bc)
    ffmpeg -y -ss "$current_offset" -i "$file_path" -vframes 1 -q:v 2 -f image2 -y -loglevel error "$output_directory/$input_file$(printf "%03d" $i).png"
    echo -ne "Extracted $i out of $number_of_frames frames.\r"
done

echo -e "\nExtraction complete. Frames saved in $output_directory."
