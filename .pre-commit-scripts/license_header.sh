#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

# license header
HEADER=$(
	cat <<'EOF'
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

EOF
)

# Compare two strings, returns true if both a are non-empty and different
safe_are_different() {
	[[ -z "$1" || -z "$2" ]] && return 1
	! diff <(echo "$1") <(echo "$2") >/dev/null
}

# Add licence header to python files
prepend_py() {
	if [[ "$(basename "$1")" == "__init__.py" ]]; then
		return
	fi
	FILE_HEADER=$(head -3 "$1")
	if safe_are_different "$FILE_HEADER" "$HEADER"; then
		echo "Editing $1 ..."
		ed -s "$1" <<EOF
1i
$HEADER

.
w
EOF
	fi
}

# yaml works just like python
prepend_yaml() {
	prepend_py "$1"
}

# add licence header to bash script, exclude the shebang
prepend_sh() {
	shebang=$(head -c 2 "$1")
	if [[ "$shebang" != "#!" ]]; then
		echo "Error: .sh scripts should start with a shebang '#!'"
		exit 1
	fi
	FILE_HEADER=$(sed -n '2,4p' "$1")
	if safe_are_different "$FILE_HEADER" "$HEADER"; then
		echo "Editing $1 ..."
		ed -s "$1" <<EOF
2i
$HEADER

.
w
EOF
	fi
}

# add header to all supported files
prepend_file() {
	FILE="$1"
	if [[ "$FILE" = *.py ]]; then
		prepend_py "$FILE"
	elif [[ "$FILE" = *.yaml ]]; then
		prepend_yaml "$FILE"
	elif [[ "$FILE" = *.sh ]]; then
		prepend_sh "$FILE"
	fi
}

for f in "$@"; do
	prepend_file "./$f"
done
