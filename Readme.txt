How to run:

# Make an environment:
python -m venv env
env\Scripts\activate

# Download Eigen v 3.4.0 from "https://eigen.tuxfamily.org/index.php":

# Extract the folder in "C:\Libraries\Eigen-3.4.0"

# Install the requirements:
pip install -r requirements.txt

# Update pip:
python.exe -m pip install --upgrade pip

# Build:
python setup.py build_ext --inplace

# Run:
python main.py