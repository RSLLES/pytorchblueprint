#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
set -euo pipefail

display_help_and_exit() {
	echo "Usage: ./multinode.sh --num-nodes=N --node-rank=R --devices=D [--main-address=<ip> | --main-address-file=<path>] [--rdzv-timeout=<seconds>] -- <SCRIPT> [ARGS...]" >&2
	exit 1
}

NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: $NODE_IP"

NUM_NODES=""
NODE_RANK=""
DEVICES=""
MAIN_ADDRESS=""
MAIN_ADDRESS_FILE=""
RDZV_TIMEOUT="1800"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--num-nodes=*) NUM_NODES="${1#*=}" ;;
	--node-rank=*) NODE_RANK="${1#*=}" ;;
	--devices=*) DEVICES="${1#*=}" ;;
	--main-address=*) MAIN_ADDRESS="${1#*=}" ;;
	--main-address-file=*) MAIN_ADDRESS_FILE="${1#*=}" ;;
	--rdzv-timeout=*) RDZV_TIMEOUT="${1#*=}" ;;
	--) shift && break ;;
	*) display_help_and_exit ;;
	esac
	shift
done

if [[ -z "$NUM_NODES" || -z "$NODE_RANK" || -z "$DEVICES" || $# -eq 0 ]]; then
	display_help_and_exit
fi

if [[ -n "$MAIN_ADDRESS_FILE" ]]; then
	MAIN_ADDRESS=$(grep -m1 -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' "$MAIN_ADDRESS_FILE") || {
		echo "Error: no IPv4 address in $MAIN_ADDRESS_FILE" >&2
		exit 1
	}
fi

if [[ -z "$MAIN_ADDRESS" ]]; then
	if [[ "$NODE_RANK" != 0 ]]; then
		echo "Error: --main-address or --main-address-file is required for node-rank > 0" >&2
		exit 1
	fi
	MAIN_ADDRESS="$NODE_IP"
fi

# torchrun reads PET_<ARG> as the default for flags fabric's CLI does not forward
export PET_RDZV_CONF="timeout=$RDZV_TIMEOUT"

exec uv run fabric run \
	--devices="$DEVICES" \
	--num-nodes="$NUM_NODES" \
	--node-rank="$NODE_RANK" \
	--main-address="$MAIN_ADDRESS" \
	--main-port=29175 \
	"$@"
