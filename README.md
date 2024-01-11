# Driver-Drowsiness-Detection-System
The Drowsiness Detection System is an intelligent safety application designed to reduce road accidents caused by driver fatigue. Using computer vision, the system employs a Convolutional Neural Network (CNN) trained to recognize facial features, specifically focusing on the eyes.

## Introduction & Problem
The Drowsiness Detection System addresses the critical issue of road accidents caused by driver fatigue. Fatigue-related accidents contribute significantly to road safety concerns, emphasizing the need for a proactive solution. The system employs computer vision techniques to monitor facial features, focusing particularly on the eyes, to detect signs of drowsiness. By utilizing a Convolutional Neural Network (CNN) model, the system aims to provide a robust and intelligent means of preventing accidents resulting from driver fatigue.

## Paradigms
The project embraces the following paradigms:
1) Computer Vision: Utilizing OpenCV for real-time video capture and image processing to monitor the driver's face.
2) Machine Learning: Implementing a Convolutional Neural Network (CNN) model through Keras for the classification of the driver's eye state.
3) Proactive Safety Measures: Employing the system to trigger alarms when signs of drowsiness are detected, acting as a proactive safety mechanism.

## Model & Explanation (CCN)
A CNN can have multiple layers, each of which learns to detect the different features of an input image. A filter or kernel is applied to each image to produce an output that gets progressively better and more detailed after each layer. In the lower layers, the filters can start as simple features.
At each successive layer, the filters increase in complexity to check and identify features that uniquely represent the input object. Thus, the output of each convolved image -- the partially recognized image after each layer -- becomes the input for the next layer. In the last layer, which is an FC layer, the CNN recognizes the image or the object it represents.
With convolution, the input image goes through a set of these filters. As each filter activates certain features from the image, it does its work and passes on its output to the filter in the next layer. Each layer learns to identify different features and the operations end up being repeated for dozens, hundreds or even thousands of layers. Finally, all the image data progressing through the CNN's multiple layers allow the CNN to identify the entire object.
The heart of the Drowsiness Detection System lies in the Convolutional Neural Network (CNN) model. The model is trained to recognize and classify facial features, particularly the state of the driver's eyes as either open or closed. This training is crucial for the system to accurately identify signs of drowsiness. The CNN model processes each frame of the live video feed, making real-time predictions and triggering alarms when prolonged eye closure is detected, indicating potential driver fatigue.

## Dataset Used
https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

## Libraries Used & Overview
The project leverages several libraries to achieve its objectives:
1) OpenCV: Used for real-time video capture and image processing, facilitating the monitoring of the driver's face.
2) Keras: Employed for building and training the Convolutional Neural Network (CNN) model to classify the state of the driver's eyes.
3) NumPy: Utilized for numerical operations, providing efficient data handling and manipulation.
4) Pandas: Used for data analysis and manipulation, ensuring effective handling of information.
5) OS: Enables interaction with the operating system, facilitating file and directory operations.
6) Pygame: Utilized for playing audio alarms, alerting the driver when signs of drowsiness are detected.

## Output
![image](https://github.com/mzainxo/Driver-Drowsiness-Detection-System/assets/120658271/c406701f-42b2-423f-b2d7-5f00e8984a28)
![image](https://github.com/mzainxo/Driver-Drowsiness-Detection-System/assets/120658271/09802fd7-e07e-4454-bdae-ef016d55dada)

## Interfaces
The system interfaces with the following components:
1) Live Video Feed: Captured in real-time through OpenCV to monitor the driver's face.
2) Convolutional Neural Network (CNN): Implemented using Keras to process and classify the state of the driver's eyes.
3) Alarm System: Utilizing Pygame to generate audio alarms, alerting the driver when drowsiness is detected.
