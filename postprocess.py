from __future__ import print_function

def postprocess(files):
    # Remove erroneous #line directives since we aren't including them deliberately
    # and they cause compilation failures on Windows.
    for cpp_file in files:
        print('Postprocessing {}'.format(cpp_file))
        with open(cpp_file, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if not line.startswith('#line')]
        with open(cpp_file, 'w') as f:
            f.writelines(lines)

if __name__ == '__main__':
    from sys import argv
    postprocess(argv[1:])
