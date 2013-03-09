import re
import subprocess
import sys

def demangle(mangled):
    cmd = ['echo %s | c++filt' % mangled]
    return subprocess.check_output(cmd, shell=True)

def process_report(fn, filt):
    current_function = ''
    function_re = re.compile(r'Compiling entry function \'(\w+)\'')
    register_re = re.compile(r'Used (\d+) registers')
    f = open(fn, 'r')
    for line in f:
        match = re.search(function_re, line)
        if match:
            current_function = demangle(match.group(1))
        match = re.search(register_re, line)
        if match:
            if re.search(filt, current_function):
                print(current_function + ': ' + match.group(1))
    f.close()

if __name__ == '__main__':
    filt = re.compile(sys.argv[2])
    process_report(sys.argv[1], filt)
    
