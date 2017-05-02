#!/usr/bin/env python

from subprocess import call, Popen, PIPE

# Install Anaconda on all nodes in cluster, set as PYSPARK_DEFAULT. (Easiest way to get NLTK running. Could have also passed nltk files.)
call('wget -P /home/anaconda2/ https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh', shell=True)

# Install Anaconda in /home/anaconda2/
call('bash /home/anaconda2/Anaconda2-4.3.1-Linux-x86_64.sh -b -f -p /home/anaconda2/', shell=True)

# Copy the nltk_script to local in order to download the NLTK library on each node.
call('gsutil cp gs://dataproc-04d7eda2-db56-484f-aba4-5db51f8b3d84-us/initialization_actions/nltk_download.py /home/install_scripts/nltk_download.py', shell=True)
call('/home/anaconda2/bin/python /home/install_scripts/nltk_download.py', shell=True)

# add Spark variable designating Anaconda Python executable as the default on driver
with open('/etc/spark/conf/spark-env.sh', 'ab') as f:
    f.write('PYSPARK_PYTHON=/home/anaconda2/bin/python' + '\n')

