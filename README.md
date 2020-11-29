# Neural-Networks-and-Deep-Learning-Models

**Tools Used**
- Python
- [TensorFlow](https://playground.tensorflow.org)  
```
# Installs latest version of TensorFlow 2.X 
pip install --upgrade tensorflow
```

## Overview

A Neural Network is a powerful machine learning technique that is modelled after neurons in the brain. Neural networks can rival the performance of the most robust statistical algorithms without having to worry about *any* statistical theory. Because of this, neural networks are an in-demand skill for any data scientist. Big tech companies use an advanced form of neural networks called **deep neural networks** to analyze images and natural language processing datasets. Retailers like Amazon and Apple are using neural networks to classify their consumers to provide targeted marketing as well as behavior training for robotic control systems. Due to the ease of implementation, neural networks also can be used by small businesses and for personal use to make more cost-effective decisions on investing and purchasing business materials. Neural networks are scalable and effective—it is no wonder why they are so popular.In this repository we will explore how neural networks are designed and how effective they can be using the **TensorFlow** platform in Python. With neural networks wer can combine the performance of multiple statistical and machine learning models with minimal effort. In fact, more time is spent preprocessing the data to be compatible with the model than spent coding the neural network model, which can be just a few lines of code.  

In this project we work with a mock company, Alphabet Soup, a foundation dedicated to supporting organizations that protect the environment, improve people's well-being, and unify the world. This company has raised and donated a great sum of money to invest in life saving technologies and organized re-forestation groups around the world. Our task will be to analyze the impact of each donation and vet potential recepients. This helps ensure that the foundation's money is being used effectively. Unfortunately, not every dollar the foundation donates is impactful. Sometimes another organization may recieve funds and disapear. As a result, we must work as data scientists to predict which organizations are worth donating to and which are too high risk. This problem seems too complex for statistical and machine learning models we have used. Instead, we will design and train a deep neural network which will evaluate all types of input data and produce a clear decision making result.


## Notes

### Neural Networks  
Neural Networks, or artificial neural networks, are a set of algorithms that are modeled after the human brain. They are an advanced for of machine learning that recognizes patterns and features in input data and procides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the finals layer, which returns either a numerical result, or an encoded categorical result. A neural network may be used to create a classification algorithm that determines if an input belongs to one category versus another. Alternatively, neural network models can behave like a regression model, where a dependant variable can be predicted from independent input variables. Therefore, neural networks are seen as an alternative to many models, such as random forestm or multiple linear regession.  
There are a number of advantages to using a neural network instead of a traditional statistical or machine learning model. For instance, neural networks are effective at detecting complex, nonlinear relationships. Additionally, neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data. The two biggest disadvantages to using a neural network model are that the layers of neurons are often too complex to dissect and understand (creating a black box problem), and neural networks are prone to overfitting (characterizing the training data so well that it does not generalize to test data effectively). However, both of the disadvantages can be mitigated and accounted for.  
Neural networks work by linking together neurons and producing a clear quantitative output. But if each neuron has its own output, how does the neural network combine each output into a single classifier or regression model? The answer is an **activation function.** The activation function is a mathematical function applied to the end of each "neuron" (or each individual perceptron model) that transforms the output to a quantitative value. This quantitative output is used as an input value for other layers in the neural network model. There are a wide variety of activation functions that can be used for many specific purposes.

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

### Increasing the Performance of Neural Network Models  
Adding neurons to a neural network can help to generate a well-performing model faster than using a single-neuron, single-layer neural network. In fact, there are multiple advantages to adding neurons to our hidden layer in a neural network. Some of the advantages follow:  
- There is a distributed effort to find optimal weights—faster.
- Each neuron can focus on different features to identify nonlinear effects—smarter.
- It is less likely to fixate on complex variables—more robust.  

So, if adding more neurons to our neural network model increases the performance, why wouldn't we always use the maximum number of neurons? There are two main reasons to limit the number of neurons in a neural network model: overfitting and computation resources. Similar to other machine learning algorithms, neural networks are susceptible to overfitting where the model fits the training data too well. As a result of overfitting, the neural network will not generalize well and won't be able to classify new data correctly. Additionally, a neural network model with a large number of neurons requires equally large training dataset—training a large neural network requires more data, more epochs, and more time. Therefore, it is important that a neural network model has an appropriate number of neurons to match the size of the data, the complexity of the problem, and the amount of input neurons. A good rule of thumb for a basic neural network is to have two to three times the amount of neurons in the hidden layer as the number of inputs.

As with all machine learning algorithms, neural networks are not perfect and will often underperform using a basic implementation. When a neural network model does not meet performance expectations, it is usually due to one of two causes: inadequate or inappropriate model design for a given dataset, or insufficient or ineffective training data. Although collecting more training/test data is almost always beneficial, it may be impossible due to budget or logistical limitations. Therefore, the most straightforward means of improving neural network performance is tweaking the model design and parameters. When it comes to tweaking a neural network model, a little can go a long way. If we tweak too many design aspects and parameters at once, we can cause a model to become less effective without a means of understanding why. To avoid trapping ourselves in endless optimization iterations, we can use characteristics of our input data to determine what parameters should be changed.

There are a few means of optimizing a neural network:  
- Check out your input dataset.  
Before you start down the endless journey of model optimization, it is always a good idea to check the input data and ensure that there are no variables or set of outliers that are causing the model to be confused. Although neural networks are tolerant of noisy characteristics in a dataset, neural networks can learn bad habits (like the brain does). Even if we standardize and scale our numerical variables, too many outliers in a single variable can lead to performance issues.

- Add more neurons to a hidden layer and additional hidden layers.  
Instead of adding more neurons, we could change the structure of the model by adding additional hidden layers, which allows neurons to train on activated input values, instead of looking at new training data. Therefore, a neural network with multiple layers can identify nonlinear characteristics of the input data without requiring more input data. This concept of a multiple-layered neural network is known as a **deep learning neural network.** 

- Use a different activation function for the hidden layers.  
Another strategy to increase performance of a neural network is to change the activation function used across hidden layers. Depending on the shape and dimensionality of the input data, one activation function may focus on specific characteristics of the input values, while another activation function may focus on others.

- Add additional epochs to the training regimen.  
If your model still requires optimizations and tweaking to meet desired performance, you can increase the number of epochs, or training iterations. As the number of epochs increases, so does the amount of information provided to each neuron. By providing each neuron more information from the input data, the neurons are more likely to apply more effective weight coefficients. Adding more epochs to the training parameters is not a perfect solution—if the model produces weight coefficients that are too effective, there is an increased risk of model overfitting. Therefore, models should be tested and evaluated each time the number of epochs are increased to reduce the risk of overfitting.  

As with all machine learning models, creating an ideal classification or regression model is part mathematics and part art. As we design more and more models, optimizing and fine-tuning becomes less trial and error and more pattern recognition.
