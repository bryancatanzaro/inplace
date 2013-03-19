from __future__ import print_function
import os.path

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

#Parallelize the build maximally
import multiprocessing
n_jobs = multiprocessing.cpu_count()
SetOption('num_jobs', n_jobs)

    
#Add CUDA Lib
env.Append(LIBS=['cudart'])

#Add OpenMP
env.Append(CCFLAGS=['-fopenmp'])
env.Append(LIBS='gomp')

#Add stdc++
env.Append(LIBS='stdc++')

Export('env')

#Build tests
SConscript(os.path.join('src', 'SConscript'))

