.PHONY: clean data lint requirements download_data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = python3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Install Python dependencies using pip
requirements:
	pip install -r requirements.txt

# Make dataset
data:
	$(PYTHON_INTERPRETER) src/data/download_data.py
	$(PYTHON_INTERPRETER) src/data/make_data.py 40 ./data/gtsrb_train ./data

# Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

# Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .