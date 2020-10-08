#!/bin/bash

# Assume that the current directory is "share_random_forest".
# This upgrades python to 3.7 if necessary
echo "Create a virtual environment with the necessary dependencies and activate it? (Reply y or n)"
read;
if [ "$REPLY" = "n" ]; then
	echo 'exit'
elif [ "$REPLY" = "y" ]; then
	sudo yum install python37
	python3 -m pip install --upgrade --user pip
	python3 -m pip install --user virtualenv
	virtualenv -p python37 rfenv
else
	echo "That is not a valid reply. Please type 'y' or 'n'."
fi
# echo "The virtual environment refenv has been created in the current directory."
source rfenv/bin/activate
pip install -r requirements.txt
python rf_example.py

