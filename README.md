======================================
SemEval 2016 Task-1 Text-Similarity 
======================================




Execute shell script 'run.sh' to run the task and to see
evaluation results for all three training file. 


Currently unsupervised methods(PCA, Doc2Vec etc.) are being
used for data representation and cosine similarity scoring
scaled between 0 and 5. 


5 for complete similarity and 0 for complete dissimilarity.


Evaluation script(a perl script) takes these similarity scores and returns 
Pearson Correlation score for each input file(between -1 to 1,
higher is better).



Pearson Correlation:

https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient


For more information about the SemEval task:

http://alt.qcri.org/semeval2016/task1/



Requirements:

Apart from what listed in requirements.txt, you also
need perl on your system with following modules:


Number::Format

Statistics::Basic


Refer this following thread to install above perl modules(use 'sudo' in case permission denied to create
new files in perl directory):

http://stackoverflow.com/questions/17719894/compiling-statisticsbasic-on-strawberry-failed