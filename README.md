DyND-Python
===========

TravisCI: [![Build Status](https://api.travis-ci.org/libdynd/dynd-python.svg?branch=master)](https://travis-ci.org/libdynd/dynd-python)
AppVeyor: [![Build Status](https://ci.appveyor.com/api/projects/status/cv2bnq3oghe4nqnj/branch/master?svg=true)](https://ci.appveyor.com/project/libdynd/dynd-python/branch/master)

DyND-Python, a component of [the Blaze project](http://blaze.pydata.org/),
is the Python exposure of [the DyND dynamic multi-dimensional array library](http://libdynd.org).

To discuss the development of this library, subscribe to the
[LibDyND Development List](https://groups.google.com/forum/#!forum/libdynd-dev).

Python versions 2.7, 3.4 and 3.5 are supported.

http://libdynd.org

https://github.com/libdynd/libdynd

https://github.com/libdynd/dynd-python

Trying Out DyND
---------------

The easiest way to try it out is through the Anaconda
Python distribution. The latest release of Anaconda includes
a version of DyND.

http://continuum.io/downloads

For trying the latest updates, there is also an automated
build configured which tracks the latest git master. When
all the tests pass, it uploads conda packages to the anaconda.org
channel "dynd/channel/dev". To get these versions, you
can run the following command.

```
conda install dynd-python --channel dynd/channel/dev
```

It may work best to install development versions of DyND into an
environment instead of the main Anaconda directory.
You can do this with a command like:

```
C:\>conda create -n dynd-env python=3.3 dynd-python --channel dynd/channel/dev
```

Developing DyND
---------------

See the [build and install instructions](BUILD_INSTALL.md) for details on
building the software for environments not supported by Anaconda, or
if you would like to modify or contribute to the project.
