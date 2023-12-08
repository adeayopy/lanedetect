## Lane detection script using computer vision

# Usage

Clone repo, create virtual environment and install dependencies using ```pip install -r requirements.txt```

The code currently works for white lane detection. Since crops rows are green, this has to be changed. The threshold for the canny edge detection should be changed. Currently, the lower and upper bounds are 100 and 200 respectively. 

Perhaps, the lane is not within the view, change the  ```region_of_interest_vertices``` 

Other parameters to look at are the arguments of the hough transformation 