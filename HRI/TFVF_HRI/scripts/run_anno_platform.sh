#!/bin/bash
cd $(dirname $0)
cd ../anno_platform
python -m http.server
