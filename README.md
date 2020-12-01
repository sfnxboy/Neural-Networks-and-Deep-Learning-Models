# Neural-Networks-and-Deep-Learning-Models

**Tools Used**
- Python
- [TensorFlow](https://playground.tensorflow.org)  
```
# Installs latest version of TensorFlow 2.X 
pip install --upgrade tensorflow
```

A Neural Network is a powerful machine learning technique that is modelled after neurons in the brain. Neural networks can rival the performance of the most robust statistical algorithms without having to worry about *any* statistical theory. Because of this, neural networks are an in-demand skill for any data scientist. Big tech companies use an advanced form of neural networks called **deep neural networks** to analyze images and natural language processing datasets. Retailers like Amazon and Apple are using neural networks to classify their consumers to provide targeted marketing as well as behavior training for robotic control systems. Due to the ease of implementation, neural networks also can be used by small businesses and for personal use to make more cost-effective decisions on investing and purchasing business materials. Neural networks are scalable and effective—it is no wonder why they are so popular.In this repository we will explore how neural networks are designed and how effective they can be using the **TensorFlow** platform in Python. With neural networks wer can combine the performance of multiple statistical and machine learning models with minimal effort. In fact, more time is spent preprocessing the data to be compatible with the model than spent coding the neural network model, which can be just a few lines of code.  

In this project we work with a mock company, Alphabet Soup, a foundation dedicated to supporting organizations that protect the environment, improve people's well-being, and unify the world. This company has raised and donated a great sum of money to invest in life saving technologies and organized re-forestation groups around the world. Our task will be to analyze the impact of each donation and vet potential recepients. This helps ensure that the foundation's money is being used effectively. Unfortunately, not every dollar the foundation donates is impactful. Sometimes another organization may recieve funds and disapear. As a result, we must work as data scientists to predict which organizations are worth donating to and which are too high risk. This problem seems too complex for statistical and machine learning models we have used. Instead, we will design and train a deep neural network which will evaluate all types of input data and produce a clear decision making result.


## Notes

### Neural Networks  
Neural Networks, or artificial neural networks, are a set of algorithms that are modeled after the human brain. THey are an advanced for of machine learning that recognizes patterns and features in input data and procides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the finals layer, which returns either a numerical result, or an encoded categorical result. A neural network may be used to create a classification algorithm that determines if an input belongs to one category versus another. Alternatively, neural network models can behave like a regression model, where a dependant variable can be predicted from independent input variables. Therefore, neural networks are seen as an alternative to many models, such as random forestm or multiple linear regession.  
There are a number of advantages to using a neural network instead of a traditional statistical or machine learning model. For instance, neural networks are effective at detecting complex, nonlinear relationships. Additionally, neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data. The two biggest disadvantages to using a neural network model are that the layers of neurons are often too complex to dissect and understand (creating a black box problem), and neural networks are prone to overfitting (characterizing the training data so well that it does not generalize to test data effectively). However, both of the disadvantages can be mitigated and accounted for.  
Neural networks work by linking together neurons and producing a clear quantitative output. But if each neuron has its own output, how does the neural network combine each output into a single classifier or regression model? The answer is an **activation function.** The activation function is a mathematical function applied to the end of each "neuron" (or each individual perceptron model) that transforms the output to a quantitative value. This quantitative output is used as an input value for other layers in the neural network model. There are a wide variety of activation functions that can be used for many specific purposes. Neural networks (and especially deep neural networks) thrive in large datasets. Datasets with thousands of data points, or datasets with complex features, may overwhelm the logistic regression model, while a deep learning model can evaluate every interaction within and across neurons.

### Process  
1.	Import dependencies.
2.	Import the input dataset.
3.	Generate categorical variable list.
4.	Create a OneHotEncoder instance.
5.	Fit and transform the OneHotEncoder.
6.	Add the encoded variable names to the DataFrame.
7.	Merge one-hot encoded features and drop the originals.
8.	Split the preprocessed data into features and target arrays.
9.	Split the preprocessed data into training and testing dataset.
10.	Create a StandardScaler instance.
11.	Fit the StandardScaler.
12.	Scale the data.
13.	Define the model.
14.	Add first and second hidden layers.
15.	Add the output layer.
16.	Check the structure of the model.

### The Perceptron  
The perceptron model, pioneered in the 1950's by Frank Rosenblatt, is a single neural network unit, and it mimics a biological neuron by recieving input data, weighing the information, and producing a clear output. The perceptron model is supervised learning because we provide the model of our input and output information. It is designed to produce a discrete classification model and to learn from the input data to improve classifications as more data is analyzed.The perceptron model has four major components:  
- **input values,** typically labelled as x or 𝝌 (chi)
- A **weight coefficient** for each input value, typically labelled as w or ⍵ (omega).
- **Bias,** a constant value added to the input the influence the final decision, typically labelled as **w0**. In other words, no matter how many inputs we have, there will always be an additional value to "stir the pot."
- A **net summary function** that aggregates all weighted inputs, in this case a weighted summation:  
![image](https://user-images.githubusercontent.com/68082808/100526248-0e9d2580-3195-11eb-94b2-1f22aec1c081.png)  

Perceptrons are capable of classifying datasets with many dimensions; however, the perceptron model is most commonly used to separate data into two groups (also known as a linear binary classifier). In other words, the perceptron algorithm works to classify two groups that can be separated using a linear equation (also known as linearly separable). This may not prove to be useful in every scenario as not all datasets are linearly seprable, but may be seprable in other manners. Say we have a 2 dimentional set of datapoints, the model will use perceptron model training again and again until one of three conditions are met:  
* The perceptron model exceeds a predetermined performance threshold, determined by the designer before training. In machine learning this is quantified by minimizing the loss metric.
* The perceptron model training performs a set number of iterations, determined by the designer before training.
* The perceptron model is stopped or encounters an error during training.

At first glance, the perceptron model is very similar to other classification and regression models; however, the power of the perceptron model comes from its ability to handle multidimensional data and interactivity with other perceptron models. As more multidimensional perceptrons are meshed together and layered, a new, more powerful classification and regression algorithm emerges—the neural network.

### Models  
![image](https://user-images.githubusercontent.com/68082808/100649864-108bf380-3311-11eb-8d4c-14f06fc6e92c.png)  

A data scientist must not just think critically about how a particular model works, but also about if it is the best model for the dataset. There are pros and cons to every model, and some fit certain data sets. Contrary to what you may believe, neural networks are not the ultimate solution to all data science problems. As shown in the figure above, there are trade-offs to using the new and popular neural network (and deep learning) models over their older, often more lightweight statistics and machine learning counterparts.

**Logistic Regression vs Basic Neural Network**  
A logistic regression model is a classification algorithm that can analyze continuous and categorical variables. Using a combination of input variables, logistic regression predicts the probability of the input data belonging to one of two groups. If the probability is above a predetermined cutoff, the sample is assigned to the first group, otherwise it is assigned to the second. Simply, logistic regression is a statistical model that mathematically determines its probability of belonging to one of two groups. At the heart of the logistic regression model is the sigmoid curve, which is used to produce the probability (between 0 and 1) of the input data belonging to the first group. This sigmoid curve is the exact same curve used in the sigmoid activation function of a neural network. In fact, a basic neural network using the sigmoid activation function is effectively a logistic regression model:  
![image](https://user-images.githubusercontent.com/68082808/100650496-03bbcf80-3312-11eb-8bfb-ed8442302854.png)  

In [this logistic regression neural network project](https://github.com/sfnxboy/Neural-Networks-and-Deep-Learning-Models/blob/main/LogisticRegression_NeuralNet.ipynb) where we try to determine if a patient has diabetes based on their other conditions. This dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and contains the patient information of 786 women. It is used as a real-world practice dataset to build a predictive diagnostic model. We first build a basic logistic regression model, which proves to have an accuracy of `0.729`. Aftwerwards we build, compile and evaluate the basic neural network model. In the logistic regression model we the program runs for 200 training iterations, however the neural network will run for a maximum of 50 epochs to limit the risk of overfitting the model. The basic neural network model failed to breach an accuracy of 80%. There are a number of reasons why the models could not achieve 80% accuracy. Perhaps the input data was insufficient, such that thee was not enough data points and too few features. Maybe both models are lacking optimization in terms of parameters, structure, and weights. It is also a possibility that the features in the input data confuse the model.

**SVM vs Deep Learning Model**
SVMs are supervised learning models that analyze data used for regression and classification. More specifically, SVMs try to calculate a geometric hyperplane that maximizes the distance between the closest data point of both groups. If we only compare binary classification problems, SVMs have an advantage over neural network and deep learning models:

- Neural networks and deep learning models will often converge on a local minima. In other words, these models will often focus on a specific trend in the data and could miss the "bigger picture."
- SVMs are less prone to overfitting because they are trying to maximize the distance, rather than encompass all data within a boundary.  

Despite these advantages, SVMs are limited in their potential and can still miss critical features and high-dimensionality relationships that a well-trained deep learning model could find. However, in many straightforward binary classification problems, SVMs will outperform the basic neural network, and even deep learning models with ease. Looking at the [output of our SVM model](https://github.com/sfnxboy/Neural-Networks-and-Deep-Learning-Models/blob/main/SVM_DeepLearning.ipynb), the model was able to correctly predict the customers who subscribed roughly 87% of the time, which is a respectable first-pass model. After training a deep learning model with a maximum of 50 epochs, one can observer that the two models both achieved a predictive accuracy of around 87%.  The only noticeable difference between the two models is implementation—the amount of code required to build and train the SVM is notably less than the comparable deep learning model. As a result, many data scientists will prefer to use SVMs by default, then turn to deep learning models, as needed.

**Random Forest vs Deep Learning Model**  
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model. Structurally speaking, random forest models are very similar to their neural network counterparts. Random forest models have been a staple in machine learning algorithms for many years due to their robustness and scalability. Both output and feature selection of random forest models are easy to interpret, and they can easily handle outliers and nonlinear data. If random forest models are fairly robust and clear, why would you want to replace them with a neural network? The answer depends on the type and complexity of the entire dataset. First and foremost, random forest models will only handle tabular data, so data such as images or natural language data cannot be used in a random forest without heavy modifications to the data. Neural networks can handle all sorts of data types and structures in raw format or with general transformations (such as converting categorical data). In the following [implementation]() we will compare how a random forest model and deep neural network perform on this bank loan status dataset, which contains about 36,000 data points, and 16 features.

Again, if we compare both model's predictive accuracy, their output is very similar. Both the random forest and deep learning models were able to predict correctly whether or not a loan will be repaid over 80% of the time. Although their predictive performance was comparable, their implementation and training times were not—the random forest classifier was able to train on the large dataset and predict values in seconds, while the deep learning model required a couple minutes to train on the tens of thousands of data points. In other words, the random forest model is able to achieve comparable predictive accuracy on large tabular data with less code and faster performance. The ultimate decision of whether to use a random forest versus a neural network comes down to preference. However, if your dataset is tabular, random forest is a great place to start.

### Checkpoints
With more formal applications of neural network and deep learning models, data scientists cannot afford the time or resources to build and train a model each time they analyze data. In these cases, a trained model must be stored and accessed outside of the training environment. With TensorFlow, we have the ability to save and load neural network models at any stage, including partially trained models. When building a TensorFlow model, if we use Keras' ModelCheckpoint method, we can save the model weights after it tests a set number of data points. Then, at any point, we can reload the checkpoint weights and resume model training. Saving checkpoints while training has a number of benefits:

- We can short-circuit our training loop at any time (stop the function by pressing CTRL+C, or by pressing the stop button at the top of the notebook). This can be helpful if the model is showing signs of overfitting.
- The model is protected from computer problems (power failure, computer crash, etc.). Worst-case scenario: We would lose five epochs' worth of optimization.
- We can restore previous model weight coefficients to try and revert overfitting.

