#!/usr/bin/env bash

VENVNAME=cv101_marie

python -m venv $VENVNAME
source $VENVNAME/Scripts/activate
pip install --upgrade pip

pip install ipython
pip install jupyter

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"