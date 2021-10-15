#!/bin/bash

set -e
set -u

# change current working directory to location of this script file
# this will be where the virtual environment will be created
THIS_DIR=$(cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)
ENV_DIR=$THIS_DIR/env


# if virtual environment does not exist, create it and install requirements
if [[ ! -d "$ENV_DIR" ]]; then
    echo "Configuring Virtual Environment"
    pushd "$THIS_DIR" > /dev/null
    python3 -m venv env
    env/bin/python3 -m pip install -r requirements.txt
    pushd > /dev/null
    echo "Running Script"
fi

# run the spy python script
# forward all input arguments to python script
$ENV_DIR/bin/python3 "$THIS_DIR/svg2gcode.py" "$@"
