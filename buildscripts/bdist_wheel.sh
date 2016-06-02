# Example script for building dynd.ndt and dynd.nd wheels.

# Build dynd.ndt
$PYTHON setup.py bdist_wheel --target=ndt &&
cd build/lib* &&
printf "\ntesting build ...\n" &&
$PYTHON -m dynd.ndt.test &&
cd - &&
rm -rf build &&

# Build dynd.nd
$PYTHON setup.py bdist_wheel --target=nd &&
cd build/lib* &&
printf "\ntesting build ...\n" &&
$PYTHON -m dynd.nd.test || exit 1


