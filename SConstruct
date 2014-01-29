from __future__ import print_function
import os.path

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()


siteconf = {}
siteconf['THRUST_DIR'] = None
siteconf['CXX'] = None

if os.path.exists("siteconf.py"):
    glb = {}
    execfile("siteconf.py", glb, siteconf)
else:
    print("""
*************** siteconf.py not found ***************
We will try building anyway, but may not succeed.
Read the README for more details.
""")

        
f = open("siteconf.py", 'w')
print("""#! /usr/bin/env python
#
# Configuration file.
# Use Python syntax, e.g.:
# VARIABLE = "value"
# 
# The following information can be recorded:
#
# THRUST_DIR : Directory where Thrust include files are found.
#
# CXX : C++ compiler
#
""", file=f)


for k, v in sorted(siteconf.items()):
    if isinstance(v, str):
        v = '"' + str(v) + '"'
    print('%s = %s' % (k, v), file=f)
        
f.close()

Export('siteconf')
if siteconf['CXX']:
    env.Replace(CXX=siteconf['CXX'])
if siteconf['THRUST_DIR']:
    #Must prepend because build-env.py might have found an old system Thrust
    env.Prepend(CPPPATH=siteconf['THRUST_DIR'])

conf = Configure(env)

def check_header(header, name, var):
    if not conf.CheckCXXHeader(header):
        print("%s headers not found." % name)
        print("Once they are installed, update %s in siteconf.py" % var)
        Exit(1)

#Check for Thrust
check_header('thrust/version.h', 'Thrust', 'THRUST_DIR')
    
#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)


env.Append(CCFLAGS=['-fopenmp'])
env.Append(LIBS=['gomp'])

#Add stdc++
env.Append(LIBS=['stdc++'])

Export('env')

#Build library
inplace = SConscript(os.path.join('inplace', 'SConscript'),
                     variant_dir='build')

#test_env adds inplace
test_env = env.Clone()
test_env.Append(CPPPATH=os.path.join(os.getcwd(), 'inplace'))
test_env.Append(LIBS=inplace)

Export('test_env')

#Build tests
SConscript(os.path.join('test', 'SConscript'))

