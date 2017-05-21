# Various Suggestions in Neural Nets

I know that ML and ANN experts provide various suggestions such as choice of parameters, optimizing codes, etc one must use while designing an algorithm.
Many students follow these suggestions blindly, and I myself don't completely trust a suggestion till I try it. Here, I try for myself and show a real example of what happens when these suggestions are not followed. 

I consider various python scripts from my repo, and show the results of the algorithm before and after incorporating these suggestions in the script. 
Later, after gathering a considerable amount of suggestions and noticing the impact on the results, I will try and create a large database of results generated using different combinations of these suggestions.
 
I also hope to expand this to other datasets than CIFAR and to other applications outside Computer Vision.

(All these suggestions will later be converted into interactive Jupyter notebooks for a better view of the results and code used for the same.)

1. Considering svmMinigdReg.py, we try a suggestion on pre-processing of various methods of centering the data and not centering as well.
	1. Not centering the data:
		When the CIFAR-10 data is not centered, this is the output:
		```
		Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent
		Test accuracy is: 22.736 %
		Total execution time: 2.336 s
		```
	1. Centering the data using:
		```
		x = x - 127
		```
		The output: 
		```
		Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent
		Test accuracy is: 10.837 %
		Total execution time: 2.543 s
		```
	1. Centering data using (USED IN THE FILE):
		```
		x = x - np.mean(x)
		```
		The output is:
		```
		Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent
		Test accuracy is: 29.077 %
		Total execution time: 3.512 s
		```
	1. Centering data using the mean for each color:
		```
		sep = int((D-1)/3)
        for col in range(3):
                x[:,sep*col:sep*(col+1)] = x[:,sep*col:sep*(col+1)] - np.mean(x[:,sep*col:sep*(col+1)])
		```	
		The output is:
		```
		Classification of CIFAR-10 dataset using regularized SVM with Minibatch gradient descent
		Test accuracy is: 18.179 %
		Total execution time: 3.884 s
		```