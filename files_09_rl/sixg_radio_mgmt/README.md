# rrm-structure

Code containing basis classes for radio resource management.

## Install

- Install [pipenv](https://github.com/pypa/pipenv)
- Install dependencies using pipenv: `pipenv install`
- To access the virtual environment created, run `pipenv shell`, now all commands which you run will be performed into virtual enviroment created
- (In case you want to contribute with this repo, if not you can skip this step) Activate pre-commit hooks to use [black formatter](https://github.com/psf/black), [flake8 lint](https://gitlab.com/pycqa/flake8), [Isort references](https://github.com/timothycrosley/isort) and [Pyright type check](https://github.com/microsoft/pyright). Run `pre-commit install`. Now every time you make a commit, black formatter, flake8, isort and pyrights will make tests to verify if your code is following the [patterns](https://realpython.com/python-pep8/) (you can adapt your IDE or text editor to follow this patterns, e.g. [vs code](https://code.visualstudio.com/docs/python/python-tutorial#_next-steps)).