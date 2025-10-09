#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

### functions ###

# print usage instuctions
usage() {
	cat <<EOF
Usage: $0 -n NEW_NAME [-o OLD_NAME] [-h]

Options:
  -n, --name   The new project name (required)
  -o, --old    The old project name (default: blueprint)
  -h, --help   Display this help message
EOF
}

# substitute old name with new name in the specified file
# given a directory, parse all its files recursively that match a name pattern
substitute() {
	if [[ -d $1 ]]; then
		find "$1" -type f -name "$2" | while read -r file; do
			substitute "$file"
		done
	elif [[ -f $1 ]]; then
		sed -i "s/${oldname}/${name}/g" "$1"
	else
		echo "$1 is not valid"
		usage
		exit 1
	fi
}

### parse arguments ###
oldname="blueprint"
while [[ $# -gt 0 ]]; do
	case "$1" in
	-n | --name)
		shift
		name="$1"
		;;
	-o | --old)
		shift
		oldname="$1"
		;;
	-h | --help)
		usage
		exit 0
		;;
	-*)
		echo "Unknown option: $1" >&2
		usage
		exit 1
		;;
	*)
		echo "Unexpected argument: $1" >&2
		usage
		exit 1
		;;
	esac
	shift
done

if [[ -z ${name:-} ]]; then
	echo "Error: New project name not specified." >&2
	usage
	exit 1
fi

### main ###
echo "Renaming 'src/${oldname}' to 'src/${name}' ..."
mv "src/${oldname}" "src/${name}"

echo "Updating python files under 'scripts/${name}' ..."
substitute "scripts/${name}" "*.py"

echo "Updating python files under 'src/${name}' ..."
substitute "src/${name}" "*.py"

echo "Updating python files under 'tests/' ..."
substitute "tests/" "*.py"

echo "Updating 'pyproject.toml' ..."
substitute "pyproject.toml"
