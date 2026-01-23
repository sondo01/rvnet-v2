#!/bin/bash

# Use find to locate all .stif files in subdirectories
# -type f ensures we only look for files
# -name "*.stif" filters for the specific extension
cd DATA

find . -type f -name "*.stif" | while read -r file; do
    # Generate the new filename by replacing the extension
    new_name="${file%.stif}.fits"
    
    # Rename the file
    mv "$file" "$new_name"
    
    echo "Renamed: $file -> $new_name"
done

echo "Done!"
