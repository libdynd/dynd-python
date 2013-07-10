import subprocess
from os import unlink
from os.path import realpath, islink, isfile, isdir
import sys
import shutil
import time

def rm_rf(path):
    if islink(path) or isfile(path):
        # Note that we have to check if the destination is a link because
        # exists('/path/to/dead-link') will return False, although
        # islink('/path/to/dead-link') is True.
        unlink(path)

    elif isdir(path):
        if sys.platform == 'win32':
            subprocess.check_call(['cmd', '/c', 'rd', '/s', '/q', path])
        else:
            shutil.rmtree(path)

def main(pyversion, envdir):
    envdir = realpath(envdir)

    rm_rf(envdir)

    packages = ['cython', 'scipy', 'nose']

    # Try for 5 minutes
    for i in range(5):
        p = subprocess.Popen(['conda', 'create', '--yes', '-p', envdir,
            'python=%s' % pyversion] + packages, stderr=subprocess.PIPE,
            shell=(sys.platform == 'win32'))
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print >> sys.stderr, stderr
            if "LOCKERROR" in stderr:
                print "Conda is locked. Trying again in 60 seconds"
                print
                time.sleep(60)
            else:
                return p.returncode
        else:
            return p.returncode
    print('Giving up on conda, manual intervention required :(')
    return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
