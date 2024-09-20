# Sentiment Analysis of IMDB Reviews Using LSTM  

This repository contains a deep learning model for sentiment analysis of IMDB movie reviews using Long Short-Term Memory (LSTM) networks. The model is trained to classify reviews as positive or negative based on the text input.  

![_dc976771-a4e5-4c94-adf1-992656072555](https://github.com/user-attachments/assets/9d45840b-4ab0-4d67-a8a9-8dca0a7ae133)


## Table of Contents  

- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Results](#results)  
- [Contributing](#contributing)  
- [Acknowledgements](#acknowledgements)  

## Dataset  

The dataset used for this project is the English IMDB Reviews dataset, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset contains 50,000 movie reviews, labeled as positive or negative.

![dataset-cover](https://github.com/user-attachments/assets/cd16bf8b-2198-443f-80f7-ad920450399e)

## Preprocessing  

Before training the model, the text data undergoes several preprocessing steps to ensure better performance:  

1. **Lowercasing Text**: All text is converted to lowercase to maintain uniformity.  
   
2. **Removing HTML Tags, URLs, and Punctuations**: HTML tags are stripped from the text, and any URLs and punctuation marks are removed to clean the data.  

3. **Handling Chatwords**: Common chat abbreviations (e.g., "IMO" for "in my opinion") are replaced with their full forms to improve understanding.  

4. **Stop Words Removal**: Commonly used words that do not contribute to the sentiment (like "and", "the", etc.) are removed from the text.  

5. **Demojizing**: Emojis present in the text are converted to their textual representations using the `emoji` library.   

6. **Stemming**: Words are reduced to their root form (e.g., "running" to "run") to standardize the input.

## Model Architecture

The sentiment analysis model is built using an LSTM architecture. Below are the key components of the model:

- Embedding Layer: Converts the input words into fixed-sized dense vectors.
- LSTM Layer: Captures the temporal dependencies in the text data.
- Dense Layer: Outputs the final prediction (positive or negative sentiment).

![image](https://github.com/user-attachments/assets/bc4a50f0-4911-4052-b16f-141af7830af3)


## Training

To train the model, use the following command in your terminal or script:

```python 
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

## Evaluation

To evaluate the model's performance, you can use the following code:

```python 
loss, accuracy = model.evaluate(X_test, y_test)  
print(f'Accuracy: {accuracy * 100:.2f}%')
```

![image](https://github.com/user-attachments/assets/02890401-bc28-47c3-aa6a-0f30a5ce21f7)

## Results

After training and evaluation, the model achieved an accuracy of approximately 88% on the test set. The confusion matrix and classification report can further analyze the model's performance.

![image](https://github.com/user-attachments/assets/304c0ba5-e205-4884-a8f4-0dbce33701b4)

## Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature-branch).
Open a pull request.

## Acknowledgements

- Kaggle for the IMDB Reviews dataset.
- The NLTK library for natural language processing tasks.
- The emoji library for demojizing.
- Keras and TensorFlow for building the deep learning model.
