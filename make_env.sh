#!/bin/bash
echo "Create a conda environment? (Reply y or n)"
read;
if [ "$REPLY" = "n" ]; then
	echo 'exit'
elif [ "$REPLY" = "y" ]; then
	conda env create -f rfenv.yml -p ./rfenv
else
	echo "That is not a valid reply. Please type 'y' or 'n'."
fi
echo "The conda environment refenv has been created in the current directory."
eval $(conda shell.bash hook)
conda activate /Users/Daniel/Code/test_rf/rfenv
echo "The conda environment rfenv has been activated."