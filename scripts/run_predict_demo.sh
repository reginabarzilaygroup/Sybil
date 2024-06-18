#!/bin/bash

# Run inference on the demo data
# The output will be printed to the console


demo_scan_dir=sybil_demo_data

# Download the demo data if it doesn't exist
if [ ! -d "$demo_scan_dir" ]; then
  # Download example data
  curl -L -o sybil_example.zip "https://www.dropbox.com/scl/fi/covbvo6f547kak4em3cjd/sybil_example.zip?rlkey=7a13nhlc9uwga9x7pmtk1cf1c&dl=1"
  unzip -q sybil_example.zip
fi

# If not installed with pip, sybil-predict will not be available.
# Can use "python3 sybil/predict.py" instead.
sybil-predict \
--loglevel DEBUG \
--output-dir demo_prediction \
--return-attentions \
$demo_scan_dir
