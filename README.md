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

## AlphabetSoupCharity Report  
The program for the following report can be found [here](https://github.com/sfnxboy/Neural-Networks-and-Deep-Learning-Models/blob/main/AlphabetSoupCharity/AlphabetSoupCharity.ipynb).  

### Overview of Analysis  
With a [CSV](https://github.com/sfnxboy/Neural-Networks-and-Deep-Learning-Models/blob/main/AlphabetSoupCharity/Resources/charity_data.csv) containing more than 34,000 organizations that the mock company, Alphabet Soup, has recieved over the years, we will attempt to build a neural network model that can accurately predict what features (or characteristics) organizations have that indicate that they will be a succesful investment. Within this dataset there are a number of columns that capture the metadata about each organization, such as the following:  

- **EIN** and **NAME**—Identification columns  
- **APPLICATION_TYPE**—Alphabet Soup application type  
- **AFFILIATION**—Affiliated sector of industry  
- **CLASSIFICATION**—Government organization classification  
- **USE_CASE**—Use case for funding  
- **ORGANIZATION**—Organization type  
- **STATUS**—Active status  
- **INCOME_AMT**—Income classification  
- **SPECIAL_CONSIDERATIONS**—Special consideration for application  
- **ASK_AMT**—Funding amount requested  
- **IS_SUCCESSFUL**—Was the money used effectively  

This project consists of three portions. First we will have to preprocess the data appropriately to be used in a neural network model. Secondly, we will compile, train, and evaluate the model. Lastly, we will attempt to optimize the neural network to improve its performance while being warry of any overfitting that may occur.

### Results  
#### Data Preprocessing  

Our objective is to build a neural network model that can predict with a reasonable degree of accuracy which organizations are most likely to succeed. Given a dataset with 11 features, we can start off by removing columns that have nothing the machine may find meaningful, such as the `EIN` and `NAME` columns. The `IS_SUCCESFUL` column will be our target feature for our neural network model. Thankfully the values in that column are already numeric (values are either 1 or 0), so we will not have to encode that column.  

Still, part of the preprocessing process is encoding categorical variables into numeric values so that the machine may find meaning in those features. Sometimes a column has so many unique values, that it may be best to consider 'bucketing' the infrequent occurences into a single variable. I decide to bucket application types that appear less than 250 times into their own category. I made my decision based on the results from the `value_count()` and `.plot.density()` methods. To complete my preprocessing step I create a OneHotEncoder instance to quantify all categorical features.  

#### Compiling, Training, and Evaluating the Model  

I built my first model with 2 hidden layers, with 80 and 30 nodes respectively. The resulted in the following performance score: `loss: 0.5537 - accuracy: 0.7258`. Considering I only ran the first model with 50 epochs, I decided to change the number of epochs and only the number of epochs in the second model. This revision resulted in the following: `loss: 0.5560 - accuracy: 0.7261`. This is an insignificant difference. My next step was to add another hidden later. My hidden layers were programmed as follows:  
```
# First hidden layer
nn_new1.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1,
    input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn_new1.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn_new1.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="sigmoid"))

# Output layer
nn_new1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```  
This model had the following performance: `loss: 0.5557 - accuracy: 0.7255`. Again, an insignificant change. 

A rule of thumb is that if your test accuracy score is larger than your training accuracy score, there is a chance that your model promotes over fitting. This happens to be the case for all four models I built and ran. Nonetheless, my performance metrics could not reach a standard I was satisfied with, albiet the model was overfitting. This may be the case because there is a noisy feature that is confusing the machine, or because the data itself does not contain enough information for the machine.



## Notes

### Neural Networks  
Neural Networks, or artificial neural networks, are a set of algorithms that are modeled after the human brain. THey are an advanced for of machine learning that recognizes patterns and features in input data and procides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the finals layer, which returns either a numerical result, or an encoded categorical result. A neural network may be used to create a classification algorithm that determines if an input belongs to one category versus another. Alternatively, neural network models can behave like a regression model, where a dependant variable can be predicted from independent input variables. Therefore, neural networks are seen as an alternative to many models, such as random forestm or multiple linear regession.  
There are a number of advantages to using a neural network instead of a traditional statistical or machine learning model. For instance, neural networks are effective at detecting complex, nonlinear relationships. Additionally, neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data. The two biggest disadvantages to using a neural network model are that the layers of neurons are often too complex to dissect and understand (creating a black box problem), and neural networks are prone to overfitting (characterizing the training data so well that it does not generalize to test data effectively). However, both of the disadvantages can be mitigated and accounted for.  

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

### Basic Neural Network  
We can apply the same Scikit-learn pipeline of **model -> fit -> predict/transform** one would use for other machine learning algorithms to run a neural network model. This is generally the process:  
1.	Decide on a model, and create a model instance.
2.	Split into training and testing sets, and preprocess the data.
3.	Train/fit the training data to the model. (Note that "train" and "fit" are used interchangeably in Python libraries as well as the data field.)
4.	Use the model for predictions and transformations.  

Check out this [Basic Neural Network](https://github.com/sfnxboy/Neural-Networks-and-Deep-Learning-Models/blob/main/Basic%20Neural%20Network/Build_Basic_Neural_Network.ipynb) file as a reference. There are four big terms machine learning engineers should be familiar when evaluating a model’s performance. The **loss metric** measures how poorly a model characterizes the data after each iteration (or epoch). The **evaluation metric** measures the quality of a machine learning model, specifically accuracy for classification models and MSE (mean squared error) for regression models. For model predictive accuracy, the higher the number the better, whereas for regression models, MSE should reduce to zero. The **optimization function** shapes and molds a neural network model while it is being trained to ensure that it performs to the best of its ability. **Activation functions** are applied to each hidden layer in the model that allows the model to combine outputs from neurons into a single classifier/regression model. The activation function is a mathematical function applied to the end of each "neuron" (or each individual perceptron model) that transforms the output to a quantitative value. This quantitative output is used as an input value for other layers in the neural network model. There are a wide variety of activation functions that can be used for many specific purposes.

According to [Heaton Research](https://www.heatonresearch.com/2017/06/01/hidden-layers.html), the number of hidden neurons should be between the size of the input layer and the size of the output layer. The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. The number of hidden neurons should be less than twice the size of the input layer. These are simply rules of thumb, and it is important to remember that building a neural network model is both parts science and art.

### Activation Functions  
The activation function is a mathematical function applied to the end of each "neuron" (or each individual perceptron model) that transforms the output to a quantitative value. This quantitative output is used as an input value for other layers in the neural network model. There are a wide variety of activation functions that can be used for many specific purposes; however, most neural networks will use one of the following activation functions:  

•	Linear  
The linear function returns the sum of our weighted inputs without transformation.  

•	ReLU (Rectified Linear Unit)  
This function returns a value from 0 to infinity, so any negative input through the activation function is 0. It is the most used activation function in neural networks due to its simplifying output, but may not be appropriate for simpler models. The ReLU function is ideal for looking at positive nonlinear input data for classification or regression.  

•	Leaky ReLU
This function is an alternative to the ReLU function, whereby negative input values will return small negative values. One may consider using a leaky ReLU function instead of a ReLU function if the dataset contains nonlinear data with many negative inputs.  

•	Tanh  
This function is identified by a characteristic S curve, however it transforms the output to a range between -1 and 1. This function can be used for classification or regression models.  

•	Sigmoid  
This function is identified by a characteristic S curve. It transforms the output to a range between 0 and 1, which is ideal for binary classification.


### A Synaptic Boost
With all machine learning algorithms, neural networks are not perfect and will often underperform using a basic implementation.  When a neural network model does not meet performance expectations, it is usually due to one of two causes: inadequate or inappropriate model design for a given dataset, or insufficient or ineffective training data. Although collecting more training/test data is almost always beneficial, it may be impossible due to budget or logistical limitations. Therefore, the most straightforward means of improving neural network performance is tweaking the model design and parameters. When it comes to tweaking a neural network model, a little can go a long way. If we tweak too many design aspects and parameters at once, we can cause a model to become less effective without a means of understanding why. To avoid trapping ourselves in endless optimization iterations, we can use characteristics of our input data to determine what parameters should be changed.  
As with all machine learning models, creating an ideal classification or regression model is part mathematics and part art. There are a few means of optimizing a neural network:

•	Check out your input dataset.  
It is always a good idea to check the input data and ensure that there are no variables or set of outliers that are causing the model to be confused. Although neural networks are tolerant of noisy characteristics in a dataset, neural networks can learn bad habits (like the brain does).

•	Add more neurons to a hidden layer, or add more hidden layers.  
Instead of adding more neurons, we could change the structure of the model by adding additional hidden layers, which allows neurons to train on activated input values, instead of looking at new training data. Therefore, a neural network with multiple layers can identify nonlinear characteristics of the input data without requiring more input data. This concept of a multiple-layered neural network is known as a **deep learning neural network.**

•	Use a different activation function for the hidden layers.  
Another strategy to increase performance of a neural network is to change the activation function used across hidden layers. Depending on the shape and dimensionality of the input data, one activation function may focus on specific characteristics of the input values, while another activation function may focus on others. To experiment and optimize using an activation function, try selecting from activation functions that are slightly more complex than your current activation function. For example, if you were trying to build a regression neural network model using a wide input dataset, you might start with a tanh activation function. To optimize the regression model, try training with the ReLU activation function, or even the Leaky ReLU activation function. In most cases, it is better to try optimizing using a higher complexity activation function rather than a lower complexity activation function. Using a higher complexity activation function will assess the input data differently without any risk of censoring or ignoring lower complexity features.

•	Add additional epochs to the training regimen.  
If your model still requires optimizations and tweaking to meet desired performance, you can increase the number of epochs, or training iterations. As the number of epochs increases, so does the amount of information provided to each neuron. By providing each neuron more information from the input data, the neurons are more likely to apply more effective weight coefficients. Adding more epochs to the training parameters is not a perfect solution—if the model produces weight coefficients that are too effective, there is an increased risk of model overfitting. Therefore, models should be tested and evaluated each time the number of epochs are increased to reduce the risk of overfitting.

### Deep Neural Networks
With our basic neural networks, input data is parsed using an input layer, evaluated in a single hidden layer, then calculated in the output layer. In other words, a basic neural network is designed such that the input values are evaluated only once before they are used in a classification or regression equation. Although basic neural networks are relatively easy to conceptualize and understand, there are limitations to using a basic neural network. A basic neural network with many neurons will require more training data that other comparable statistical/machine learning models to produce an adequate model. Furthermore, basic neural networks struggle to interpret complex nonlinear numerical data, or data with many confounding factors that have hidden effects on more than one variable. Also, they are incapable of analyzing image datasets without severe data preprocessing.  

To address the limitations of the basic neural network, we can implement a more robust neural network model by adding additional hidden layers. A neural network with more than one hidden layer is known as a **deep neural network**. Deep neural networks function similarly to the basic neural network, with one major exception. The outputs of one hidden layer of neurons (that have been evaluated and transformed using an activation function) become the inputs to additional hidden layers of neurons. As a result, the next layer of neurons can evaluate higher order interactions between weighted variables and identify complex, nonlinear relationships across the entire dataset. These additional layers can observe and weight interactions between clusters of neurons across the entire dataset, which means they can identify and account for more information than any number of neurons in a single hidden layer. Although the numbers are constantly debated, many data engineers believe that even the most complex interactions can be characterized by as few as three hidden layers.  

Deep neural network models also are commonly referred to as **deep learning models** due to their ability to learn from example data, regardless of the complexity or data input type. Just like humans, deep learning models can identify patterns, determine severity, and adapt to changing input data from a wide variety of data sources. Compared to basic neural network models, which require many neurons to identify nonlinear characteristics, deep learning models only need a few neurons across a few hidden layers to identify the same nonlinear characteristics.  

In addition, deep learning models can train on images, natural language data, soundwaves, and traditional tabular data (data that fits in a table or Data Frame), all with minimal preprocessing and direction. The best feature of deep learning models is its capacity to systematically process multivariate and abstract data while simultaneously achieving performance results that can mirror or even exceed human-level performance. 

### Evaluating Neural Network Models  
```
# Evaluate the model using the test data
model_loss, model_accuracy = nn_new.evaluate(X_test_scaled,y_test,verbose=2)
model_performance = print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
model_performance

# Create a DataFrame containing training history
history_df = pd.DataFrame(fit_model.history, index=range(1,len(fit_model.history["loss"])+1))

# Plot the loss
history_df.plot(y="loss")

# Plot the accuracy
history_df.plot(y="accuracy")
```  
When training data, the model will output a degree of loss and accuracy. It is important to note that this calculation is based on the training data. The code above evaluates the loss and accuracy of the test data. It is important to note that if the training scores are greater than the evaluation scores, may indicate that there is a chance of overfitting. This applies heavily more towards the accuracy metric.
