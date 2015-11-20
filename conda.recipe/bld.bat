cd %RECIPE_DIR%
cd ..

:: Remove the build directory since incremental builds aren't currently supported.
rd /s /q build

python setup.py install || exit 1
rd /s /q %SP_DIR%\__pycache__
rd /s /q %SP_DIR%\numpy
rd /s /q %SP_DIR%\Cython
