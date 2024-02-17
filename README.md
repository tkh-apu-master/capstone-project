

Prerequisites:
1. Anaconda/Miniconda with Python 3.8
2. Jupyter Notebook

# Note(s)
1. Update the Python environment when the dependencies is updated. This is to ensure reproducibility in different machines.

```bash
pip list --format=freeze > requirements.txt
```

2. Ensure you are in the right environment
* Anaconda/Miniconda:
```bash
conda env list
```

3. Isolate the Python environment of this project using [venv](https://python.land/virtual-environments/virtualenv). I have seperated and commit the configuration files.

```bash
python -m venv capstone-project
source capstone-project/bin/activate  # On MacOS/Linux
capstone-project\Scripts\activate.bat  # On Windows
```

# Instructions

0. Create Conda environment specifically for the project, link:
https://saturncloud.io/blog/how-to-install-tensorflow-with-anaconda-on-windows/

1. Install the Python dependencies:

```bash
conda activate tensorflow
pip install -r requirements.txt
```

2. Try to click Run All in you Python notebook.

3. That's it for now.