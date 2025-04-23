#!/bin/bash

# Directory containing the .txt files
directory="./labels/"

# Loop through each .txt file in the directory
for file in "$directory"/*.txt; do
  # Use awk to count words in each line and check if any line has fewer than 5 words
  awk '{
    if (NF < 6) {
      print FILENAME, NR, "has", NF, "words"
    }
  }' "$file"
done

