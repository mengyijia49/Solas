xmake f -m release
xmake
PYTHONPATH=python python3 -m unittest discover -s test