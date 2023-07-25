**README**

1.  **PROJECT NAME:**

-   Fake News Detection Model

2.  **PROJECT OVERVIEW:**

-   This project implements a fake news detection model using the
    Multinomial Naive Bayes algorithm.

-   The goal is to classify news articles as \"FAKE\" or \"REAL\" based
    on their content.

3.  **DATASET:**

-   You can Download the dataset here:

[**https://drive.google.com/file/d/1Uv0l0gSf-gUqEa6ivFofn_E\_dfb4BDCc/view?usp=sharing**](https://drive.google.com/file/d/1Uv0l0gSf-gUqEa6ivFofn_E_dfb4BDCc/view?usp=sharing)

4.  **Dependencies:**

The following Python libraries need to be installed to run the code:

-   pandas: for data manipulation and handling the dataset.

-   nltk: for text preprocessing, including stopword removal and
    stemming.

-   sklearn: for model building and evaluation, including
    CountVectorizer and MultinomialNB.

-   matplotlib: for plotting the confusion matrix.

-   numpy: for numerical computations.

You can install these libraries using pip or conda package manager.

5.  **Usage:**

a)  Load the Dataset:

-   The dataset is loaded from a CSV file using pandas.

b)  Text Preprocessing:

-   The text in the \"title\" column of the dataset is preprocessed.

-   Special characters are removed, and the text is converted to
    lowercase.

-   Tokenization is performed, and stopwords are removed.

-   The text is stemmed to reduce words to their base form.

c)  CountVectorizer:

-   The text is converted to a numerical representation using
    CountVectorizer.

-   The CountVectorizer creates a bag of words, representing word
    frequency in each document.

d)  Model Training:

-   The Multinomial Naive Bayes algorithm is used for model training.

-   The dataset is divided into training and testing sets.

-   The model is trained on the training set.

e)  Model Evaluation:

-   The model is evaluated on the testing set.

-   Accuracy is calculated to measure the model\'s performance.

-   A confusion matrix is plotted to visualize the true and predicted
    labels.

**6) License:**

-   This project is licensed under the MIT License.

**7**) **Contact Information:**

For any questions or feedback, please contact:

-   Name -- Niket Virendra Patil

-   Email -- pniket7@gmail.com
