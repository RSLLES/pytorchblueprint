#!/bin/bash
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


mv "src/${oldname}" "src/${name}"

substitute () {
    grep -rl $1 | xargs sed -i "s/${oldname}/${name}/g"
}

substitute "src/${name}" --include="*.py" 
substitute "tests/" --include="*.py" 
substitute "pyproject.toml"
