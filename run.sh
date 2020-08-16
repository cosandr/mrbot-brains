#!/bin/bash

_data_path="/srv/containers/discord/data"

if [[ ! -d $_data_path ]]; then
    _data_path="/dresrv${_data_path}"
fi


export DATA_PATH="$_data_path"
export UPLOAD_PATH=/tmp/mrbot-brains
LISTEN_ADDRESS="localhost:7762"

if [[ ! -d $DATA_PATH ]]; then
    echo "WARNING: Data path doesn't exist $DATA_PATH"
else
    echo "DATA_PATH: $DATA_PATH"
fi

echo "UPLOAD_PATH: $UPLOAD_PATH"
echo "LISTEN_ADDRESS: $LISTEN_ADDRESS"
python launcher.py --listen-address "$LISTEN_ADDRESS"
