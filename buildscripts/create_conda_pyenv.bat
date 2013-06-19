REM $1 is the python version
REM $2 is the directory in which the conda env is created

rd /q /s %2
call C:\Anaconda\Scripts\conda create --yes -p %2 python=%1 cython=0.19 scipy nose
echo on
