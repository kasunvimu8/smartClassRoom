#smartClassRoom

Background
-----------
smart class room contain facial recognition attending system and identify the sleep student in the classroom .
we use some enhanced image processing methods to train the data as also recognise the objects.We use some machine learning  
algorithem for detection the faces and eyes directly.

In openCV this task is quite simple but if you see the output they are not accurate as much as we expected.There are several resons 
behind that.In this project we are going to address those issues using image enchancement techniques.
(by not using only the face for prediction but the eye and etc)
smart class room contain facial recognition attending system and identify the sleep student in the classroom .we use some enhanced image processing enhanced methods to train the data as also recognise the objects.  

Introduction
________________________________________

Smart class room contain facial recognition attending system and identify the sleep student in the classroom .we use some enhanced image processing methods to train the data as also recognize the objects. We use some machine learning
algorithms for detection the faces and eyes directly. In openCV this task is quite simple but if you see the output they are not accurate as much as we expected. There are several reasons behind that. In this project we are going to address those issues using image enhancement techniques. (by not using only the face for prediction but the eye and etc.) smart class room contain facial recognition attending system and identify the sleep student in the classroom .we use some enhanced image processing enhanced methods to train the data as also recognize the objects.

Identify faces  
--------------
We use algorithm called “Chaos & AdaBoost ”.
•	Firstly, this algorithm uses the image color segmentation for coarse screening on the face image.
•	Secondly, the adaptive median filtering is applied to remove noise in  the face image to improve the quality of the face image.
•	Finally, the chaotic genetic algorithm is used to optimize the AdaBoost algorithm to achieve higher detection rate and detection speed.


How this works 
--------------

In order to understand how Face Recognition works, let us first get an idea of the concept of a feature vector. Every Machine Learning algorithm takes a dataset as input and learns from this data. The algorithm goes through the data and identifies patterns in the data.
 For instance, suppose we wish to identify whose face is present in a given image, there are multiple things we can look at as a pattern:
•	Height/width of the face.
•	Height and width may not be reliable since the image could be rescaled to a smaller face. However, even after rescaling, what remains unchanged are the ratios – the ratio of height of the face to the width of the face won’t change.
•	Color of the face.
•	Width of other parts of the face like lips, nose, etc.
Clearly, there is a pattern here – different faces have different dimensions like the ones above. Similar faces have similar dimensions. The challenging part is to convert a particular face into numbers – Machine Learning algorithms only understand numbers. This numerical representation of a “face” (or an element in the training set) is termed as a feature vector. A feature vector comprises of various numbers in a specific order.
As a simple example, we can map a “face” into a feature vector which can comprise various features like:
•	Height of face (cm)
•	Width of face (cm)
•	Average color of face (R, G, B)
•	Width of lips (cm)
•	Height of nose (cm)
Essentially, given an image, we can map out various features and convert it into a feature vector like:
Height of face (cm)	Width of face (cm)	Average color of face (RGB)	Width of lips (cm)	Height of nose (cm)
23.1	15.8	(255, 224, 189)	5.2	4.4
 
So, our image is now a vector that could be represented as (23.1, 15.8, 255, 224, 189, 5.2, 4.4). 
Of course there could be countless other features that could be derived from the image (for instance, hair color, facial hair, spectacles, etc). However, for the example, let us consider just these 5 simple features.
Now, once we have encoded each image into a feature vector, the problem becomes much simpler. Clearly, when we have 2 faces (images) that represent the same person, the feature vectors derived will be quite similar. Put it the other way, the “distance” between the 2 feature vectors will be quite small.
Machine Learning can help us here with 2 things:
1.	Deriving the feature vector: it is difficult to manually list down all of the features because there are just so many. A Machine Learning algorithm can intelligently label out many of such features. For instance, a complex features could be: ratio of height of nose and width of forehead. Now it will be quite difficult for a human to list down all such “second order” features.
2.	Matching algorithms: Once the feature vectors have been obtained, a Machine Learning algorithm needs to match a new image with the set of feature vectors present in the corpus.
Now that we have a basic understanding of how Face Recognition works, let us build our own Face Recognition algorithm using some of the well-known Python libraries.
But the algorithm that we have use most of the time face classification for decide the predictions. And also we observed predictions most of the times go wrong because of the lightning conditions. Preprocessing Steps also affect for the predictions. Therefore we should make sure that these steps are maintained. Sometimes because of noise the predictions may go wrong.

Additions to the existing solutions
-----------------------------------

•	Classifications is based on basically face and eyes. (more unique features to classify the model higher the predictions accuracy )
•	Histogram equalization. (Generally pixels from all regions of the image are good)

Further Development
-------------------
As our project name says we planning to build these programs to replace the traditional attending system and build an efficient teaching systems for lectures to find out that the students are listen to the lectures or not. (If the lecture is boring lecturer can change the way of lecturing) .or rather do something to overcome those conditions. Therefore this is only the first phase of it.

