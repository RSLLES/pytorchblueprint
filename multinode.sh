#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

display_help_and_exit() {
	echo "Usage: ./multinode.sh --n_nodes=X --node_rank=X --n_devices=X [--main_address=<ip|file>] [--timeout=<seconds>] -- <COMMAND> (uv run is implicit)"
	exit 1
}

read_first_ip() {
	local path="$1"
	local ip
	ip=$(grep -m1 -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' "$path")
	if [[ -z "$ip" ]]; then
		echo "Error: no IP address found in $path" >&2
		exit 1
	fi
	echo "$ip"
}

CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: $CURRENT_IP"

N_NODES=""
NODE_RANK=""
N_DEVICES=""
MAIN_ADDRESS=""
TIMEOUT="1800"
CMD_ARGS=()

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
	--timeout=*)
		TIMEOUT="${1#*=}"
		shift
		;;
	--)
		shift
		CMD_ARGS=("$@")
		break
		;;
	*)
		display_help_and_exit
		;;
	esac
done

if [[ -z "$N_NODES" ]] || [[ -z "$NODE_RANK" ]] || [[ -z "$N_DEVICES" ]] || [[ ${#CMD_ARGS[@]} -eq 0 ]]; then
	display_help_and_exit
fi

# MAIN_ADDRESS logic
if [[ -f "$MAIN_ADDRESS" ]]; then
	MAIN_ADDRESS=$(read_first_ip "$MAIN_ADDRESS") || exit 1
	echo "Main node IP: $MAIN_ADDRESS"
fi

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

# torchrun reads PET_<ARG> as the default for flags fabric's CLI does not forward
export PET_RDZV_CONF="timeout=$TIMEOUT"

exec uv run python -m lightning.fabric.cli \
	--devices "$N_DEVICES" \
	--num_nodes "$N_NODES" \
	--main_address "$MAIN_ADDRESS" \
	--main_port 29175 \
	--node_rank "$NODE_RANK" \
	"${CMD_ARGS[@]}"
