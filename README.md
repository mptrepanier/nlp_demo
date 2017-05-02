# Instructions for Getting this Up and Running on a Dataproc Cluster:
1) Add `init_anaconda_cluster.py` and `nltk_download.py` as DataProc initialization scripts.
    - Anaconda will be installed on every cluster node and set to PySpark's default.
    - The NLTK data will be installed on every cluster node at `/home/nltk_data`.
2) Run scripts using `gcloud dataproc jobs submit pyspark --cluster clustername pythonfile.py`.

