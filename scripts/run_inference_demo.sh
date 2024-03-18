#!/bin/bash

# Run inference on the demo data
# The output will be printed to the console


demo_scan_dir=sybil_demo_data

# Download the demo data if it doesn't exist
if [ ! -d "$demo_scan_dir" ]; then
  # Download example data
  curl -L -o sybil_example.zip "https://www.dropbox.com/scl/fi/covbvo6f547kak4em3cjd/sybil_example.zip?rlkey=7a13nhlc9uwga9x7pmtk1cf1c&dl=1"
  tar -xf sybil_example.zip
fi

python3 scripts/inference.py \
--loglevel DEBUG \
--output-dir demo_prediction \
--return-attentions \
$demo_scan_dir