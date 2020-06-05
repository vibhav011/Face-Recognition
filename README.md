# Face-Recognition
This project is made by me as a part of Seasons of Code (SoC) program of IIT Bombay.<br/>
In this project I used Principal Component Analysis (PCA) to reduce the dimensionality of training set of face images which made it very efficient for comparing the test images with each category.<br/>
In the basic implementation, face_recognition.m, I simply calculated the mean squared distance and assigned it to the category from which the mean distance is minimum. This approach gave an accuracy of about 71%.<br/>
In a more complex implementation, face_recognition_nn.m, after reducing the dimensions, I trained the reduce images in a neural network and then passed the tets images through the trained neural network from classification. This increased the accuracy to about 98%.
