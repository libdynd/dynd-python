REM $1 is the python version
REM $2 is the directory in which the conda env is created

rd /q /s %2
call conda create --yes -p %2 python=%1 cython scipy
echo on
