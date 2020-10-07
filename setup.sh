#!/bin/bash
echo "Create a virtual environment with the necessary dependencies and activate it? (Reply y or n)"
read;
if [ "$REPLY" = "n" ]; then
	echo 'exit'
elif [ "$REPLY" = "y" ]; then
	virtualenv rfvenv
else
	echo "That is not a valid reply. Please type 'y' or 'n'."
fi
# echo "The virtual environment refenv has been created in the current directory."
source rfvenv/bin/activate
pip install -r requirements.txt
python play.py