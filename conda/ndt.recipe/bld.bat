cd %RECIPE_DIR%
cd ..\..\

"%PYTHON%" setup.py install --target=ndt --single-version-externally-managed --record=record.txt || exit 1

rd /s /q %SP_DIR%\__pycache__
rd /s /q %SP_DIR%\numpy
rd /s /q %SP_DIR%\Cython
