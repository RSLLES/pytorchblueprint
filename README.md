# Pytorch project blueprint

This repository contains a blueprint of a pytorch project that I use to backbone all my projects.
The goal is to propose a minimalistic, easy to understand code while being efficient and flexible.

## Deploy

Clone the project and run the `deploy.sh` script.
It will rename your project and self-delete, after what you are good to go.

## Dependencies

My projects rely on those mandatory libraries:
- pytorch: basic machine learning framework, with GPU support and autograd.
- hydra: elegantly manage dependencies in a flexible way
- fabric: flexible and efficient way of managing multi GPU settings
- pytest: to perform your unittest
- matplotlib : to plot results
- pre-commiut: to add some pre commit hooks to format the code

## Architecture

- scripts: contains entry points scripts, like training, validation, inference.
- data: contains the data used in datasets (here, empty)
- outputs: contains training runs, with both logs and checkpoints
- configs: contains hydra config files
- tests: contains unit test for different parts of your projects that are important
- src: contains the core of your project.

Inside `src` you get a pretty classic architecture.

The utils directory should contains very minimalist function all gather in a easy to understand filename.
It's better to give those files a good name so you can easily find it while searching for files in your project, than if they were called "utils.py" in their respective directory.

## Code convention
- functionnal when possible: it ease unittest. But I do not exclude OOP: avoiding it will make Pytorch very acward and it can help to encapsulated informations.
- datasets return dictionnaries. This helps with dataset that can retrieve variable numbers of elements.
- random should always be reproductible: uses generator with seeds
- class name ar CamelCase (can be long), function name use snake_case (tends to be short), variable are onlinecase (very short). 
- clean typing helps provide additionnal information. Docstring are keep as short as possible.
- formatting: imports are sorted with ruff
- linting and formatting with ruff / static check with ty

### Comments
Comments are minimalistic and follows common guidelines, see https://google.github.io/styleguide/cppguide.html#Comments

### Format
Pre-commits hooks ensure formating (with ruff) and license headers.