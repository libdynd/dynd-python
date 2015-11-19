cd %RECIPE_DIR%
cd ..

python setup.py install || exit 1
rd /s /q %SP_DIR%\__pycache__
rd /s /q %SP_DIR%\numpy
rd /s /q %SP_DIR%\Cython
