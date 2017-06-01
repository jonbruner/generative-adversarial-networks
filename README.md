# Introduction to generative adversarial networks

This repository contains code to accompany [the O'Reilly tutorial on generative adversarial networks](https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners) written by [Jon Bruner](https://github.com/jonbruner) and [Adit Deshpande](https://github.com/adeshpande3). See [the original tutorial](https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners) to run this code in a pre-built environment on O'Reilly's servers with cell-by-cell guidance, or run these files on your own machine.

There are three versions of our simple GAN model in this repository:
- **[gan-notebook.ipynb](gan-notebook.ipynb)** is identical to the interactive tutorial, available here so that you can run it on your own machine.
- **[gan-script.py](gan-script.py)** is a straightforward Python script containing code drawn directly from the tutorial, to be run from the command line. Note that it doesn't print anything when it's executed, but it does send regular updates to [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) so that you can track its progress.
- **[gan-script-fast.py](gan-script-fast.py)** is a modest refactoring of gan-script.py that runs slightly faster because more of its computations are contained in the TensorFlow graph.

## Requirements and installation
In order to run [gan-script.py](gan-script.py) or [gan-script-fast.py](gan-script-fast.py), you'll need **[TensorFlow](https://www.tensorflow.org/install/) version 1.0 or later** and [NumPy](https://docs.scipy.org/doc/numpy/user/install.html). In order to run [gan-notebook.ipynb](gan-notebook.ipynb), you'll additionally need [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html) and [matplotlib](https://matplotlib.org/).

If you've already got TensorFlow on your machine, then you've got NumPy and should be able to run the raw Python scripts.

### Installing Anaconda Python and TensorFlow
The easiest way to install TensorFlow as well as NumPy, Jupyter, and matplotlib is to start with the Anaconda Python distribution.

1. Follow the [installation instructions for Anaconda Python](https://www.continuum.io/downloads). **We recommend using Python 3.6.**

2. Follow the platform-specific [TensorFlow installation instructions](https://www.tensorflow.org/install/). Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named `tensorflow`.

3. If you aren't still inside your Conda TensorFlow environment, enter it by opening your terminal and typing
    ```bash
    source activate tensorflow
    ```

4. Download and unzip [this entire repository from GitHub](https://github.com/jonbruner/generative-adversarial-networks), either interactively, or by entering
    ```bash
    git clone https://github.com/jonbruner/generative-adversarial-networks.git
    ```

5. Use `cd` to navigate into the top directory of the repo on your machine

6. Launch Jupyter by entering
    ```bash
    jupyter notebook
    ```
    and, using your browser, navigate to the URL shown in the terminal output (usually http://localhost:8888/)
