#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

display_help_and_exit() {
	echo "Usage: ./multinode.sh --n_nodes=X --node_rank=X --n_devices=X [--main_address=X] -- <COMMAND>"
	exit 1
}

CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: $CURRENT_IP"

N_NODES=""
NODE_RANK=""
N_DEVICES=""
MAIN_ADDRESS=""
CMD_ARGS=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--n_nodes=*)
		N_NODES="${1#*=}"
		shift
		;;
	--node_rank=*)
		NODE_RANK="${1#*=}"
		shift
		;;
	--n_devices=*)
		N_DEVICES="${1#*=}"
		shift
		;;
	--main_address=*)
		MAIN_ADDRESS="${1#*=}"
		shift
		;;
	--)
		shift
		CMD_ARGS="$@"
		break
		;;
	*)
		display_help_and_exit
		;;
	esac
done

if [[ -z "$N_NODES" ]] || [[ -z "$NODE_RANK" ]] || [[ -z "$N_DEVICES" ]] || [[ -z "$CMD_ARGS" ]]; then
	display_help_and_exit
fi

# MAIN_ADDRESS logic
if [[ "$NODE_RANK" == "0" ]]; then
	if [[ -z "$MAIN_ADDRESS" ]]; then
		MAIN_ADDRESS="$CURRENT_IP"
	fi
else
	if [[ -z "$MAIN_ADDRESS" ]]; then
		echo "Error: --main_address is required for node_rank > 0"
		exit 1
	fi
fi

exec uv run python -m lightning.fabric.cli \
	--devices "$N_DEVICES" \
	--num_nodes "$N_NODES" \
	--main_address "$MAIN_ADDRESS" \
	--main_port 29175 \
	--node_rank "$NODE_RANK" \
	$CMD_ARGS
