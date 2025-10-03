# Pytorch project blueprint

This project presents the architecture I ended up using to manage my Pytorch project.
Its goal is to be easy to understand and minimalistic while staying efficient and flexible.
Overtime, I found that flat is better than nested for code.

It also contains classical architectures that you may or may not use.
I work in computer vision so I tend to use U-Net and ViT a lot, this is why there are included by default.
Feel free to delete those.

## Deploy

Clone the project and run the `deploy.sh` script.
It will rename some directory and you will be good to go.

## Dependencies

I find that those library offers a good 
- pytorch: basic machine learning framework, with GPU capacity and autograd.
- hydra: elegantly manage dependencies in a flexible way
- fabric: very souple and efficient way of managing multi gpu settings
- pytest: to perform your unittest

## Architecture

- scripts: contains entry points scripts, like training, validation, inference.
- data: contains the data used in datasets
- outputs: contains outputs of the network
- configs: contains config files for hydra
- tests: contains unit test for different parts of your projects that are important
- src/project_name: contains the core of your project. The separation is important to get import parity.

Inside source you get a pretty classic architecture.

The utils directory should contains very minimalist function all gather in a easy to understand filename.
It's better to give those files a good name so you can easily find it while searching for files in your project, than if they were called "utils.py" in their respective directory.

## Code convention
- functionnal when possible: it ease unittest. But I do not exclude OOP: avoiding it will make Pytorch very acward and it can help to encapsulated informations.
- datasets return dictionnaries. This helps with dataset that can retrieve variable numbers of elements.
- random should always be reproductible: uses generator with seeds
- class name ar CamelCase (can be long), function name use snake_case (tends to be short), variable are onlinecase (very short). 
- clean typing helps provide additionnal information. Docstring are keep as short as possible.
- 
- imports are sorted with ruff
- linting and formatting with ruff / static check with ty

### Comments
See https://google.github.io/styleguide/cppguide.html#Comments
Comments: the best code is self-documenting.
Every file should contain license boilerplate. This is done automatically.
The class comment should provide the reader with enough information to know how and when to use the class
Every function declaration should have comments immediately following it that, describe what the function does and how to use it if not obvious.
Do not state the obvious. In particular, don't literally describe what code does, unless the behavior is nonobvious to a reader who understands C++ well. Instead, provide higher-level comments that describe why the code does what it does, or make the code self-describing.
use TODO comment 