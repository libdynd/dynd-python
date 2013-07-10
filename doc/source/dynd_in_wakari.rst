====================
Using DyND in Wakari
====================

This document explains how to install the latest
dynd builds from the anaconda build channel in
wakari. If you don't already have a wakari account,
go to https://www.wakari.io/ and sign up.

Adding the Anaconda dev channel
-------------------------------

Create the file .condarc in the path ~/. You can hit the "+file" icon
in the left panel of wakari, and type ".condarc" as the filename.
The contents of the file should look something like this::

    channels:
      - http://repo.continuum.io/pkgs/dev
      - http://repo.continuum.io/pkgs/free
      - http://repo.continuum.io/pkgs/pro
      - http://repo.continuum.io/pkgs/gpl

To double check that changing the file worked, now open a terminal window
of a shell (Pick Shell, np17py27-1.5, and click +Tab), and run
the following::

    [~]$ conda info

    Current conda install:

                 platform : linux-64
    conda command version : 1.5.2
           root directory : /opt/anaconda
           default prefix : /opt/anaconda/envs/np17py27-1.5
             channel URLs : http://repo.continuum.io/pkgs/dev/linux-64/
                            http://repo.continuum.io/pkgs/free/linux-64/
                            http://repo.continuum.io/pkgs/pro/linux-64/
                            http://repo.continuum.io/pkgs/gpl/linux-64/
    environment locations : /opt/anaconda/envs
              config file : /user_home/w_mwiebe/.condarc

You should see the path to the dev channel as seen here.

Create a custom environment
---------------------------

Now create a custom environment which includes the dynd-python package
from the dev channel.::

    [~]$ conda create -n dynd python=2.7 dynd-python numpy scipy ipython matplotlib

    Package plan for creating environment at /opt/anaconda/envs/dynd:

    The following packages will be downloaded:

        dynd-python-0.3.1_9_ge2f8fa8-py27_0.tar.bz2 [http://repo.continuum.io/pkgs/dev/linux-64/]
        <snip>

    The following packages will be linked:

        package                    |  build
        -------------------------  |  ---------------
        cairo-1.12.2               |                1
        dateutil-2.1               |           py27_1
        dynd-python-0.3.1_9_ge2f8fa8  |           py27_0
        <snip>
        zlib-1.2.7                 |                0


    Proceed (y/n)?

Running an IPython shell
------------------------

From the terminals tab, choose IPython, dynd (listed under Anaconda custom),
and click +Tab. You should now be able to import dynd, run the unit tests,
and experiment with dynd from this shell. Here's an example.::

    In [1]: import dynd

    In [2]: dynd.test()
    Running unit tests for the DyND Python bindings
    Python version: 2.7.5 |Continuum Analytics, Inc.| (default, May 31 2013, 10:40:18)
    [GCC 4.1.2 20080704 (Red Hat 4.1.2-54)]
    Python prefix: /opt/anaconda/envs/dynd
    DyND-Python module: /opt/anaconda/envs/dynd/lib/python2.7/site-packages/dynd
    DyND-Python version: 0.3.1-9-ge2f8fa8
    DyND-Python git sha1: e2f8fa8fc09120372e99583fdba6dd806196b766
    LibDyND version: 0.3.1-2-g25d6d4e
    LibDyND git sha1: 25d6d4ef6ba4b1213359b0da80897c04c8d7474a
    NumPy version: 1.7.1
    ......................................................................
    ....................................
    ----------------------------------------------------------------------
    Ran 106 tests in 0.229s

    OK
    Out[2]: <unittest.runner.TextTestResult run=106 errors=0 failures=0>

    In [3]: from dynd import nd, ndt

    In [4]: nd.array([[1,2,3], [4,5]])
    Out[4]: nd.array([[1, 2, 3], [4, 5]], strided_dim<var_dim<int32>>)

Creating a Notebook
-------------------

Wakari appears to have an issue with notebooks in custom environments
at the moment, your mileage may vary.
