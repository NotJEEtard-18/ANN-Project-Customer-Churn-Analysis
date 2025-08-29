# 📊Customer Churn Prediction using Artificial Neural Networks 

This project implements a customer churn prediction model using an Artificial Neural Network (ANN) built with TensorFlow and Keras. The model is deployed as a web application using Streamlit, allowing users to input customer details and receive a churn probability prediction. 🚀

## **📋Table of Contents** 

*   **Project Overview** 
*   **Features** 
*   **Dataset** 
*   **Installation** 
    *   Prerequisites 
    *   Clone the Repository 
    *   Install Dependencies 
*   **Usage** 
    *   Running the Streamlit Application 
    *   Making Predictions
*   **Model Details** 
*   **File Structure** 
*   **Contributing** 
*   **License** 
*   **Contact** 

## **✨Project Overview** 

Customer churn is a critical problem for many businesses, as retaining existing customers is often more cost-effective than acquiring new ones. This project aims to predict whether a customer will churn (exit the bank) based on various demographic and financial attributes. An Artificial Neural Network (ANN) is trained on historical customer data to learn patterns associated with churn, and the trained model is then used to make predictions through an interactive Streamlit web interface. 🏦➡️💔

## **💡Features** 

*   **Interactive Web Application:** User-friendly interface built with Streamlit for easy input and prediction. 
*   **Real-time Predictions:** Get instant churn probability predictions based on provided customer data. 
*   **Machine Learning Model:** Utilizes a deep learning model (ANN) for accurate predictions. 
*   **Data Preprocessing:** Includes necessary data preprocessing steps (scaling, encoding) for model input. 
*   **Serialization:** Uses `pickle` to load pre-trained encoders and scalers, ensuring consistent data transformation. 

## **📁Dataset** 

The model is trained on the `Churn_Modelling.csv` dataset. This dataset contains information about bank customers, including:

*   `CreditScore`: Customer's credit score. 
*   `Geography`: Country of the customer (France, Spain, Germany). 
*   `Gender`: Gender of the customer. 
*   `Age`: Age of the customer. 
*   `Tenure`: Number of years the customer has been a bank customer. 
*   `Balance`: Customer's account balance. 
*   `NumOfProducts`: Number of bank products the customer uses. 
*   `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No). 
*   `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No). 
*   `EstimatedSalary`: Estimated salary of the customer. 
*   `Exited`: Whether the customer churned (1 = Yes, 0 = No) - *This is the target variable*. 

## **🛠️Installation** 

### Prerequisites 

Before you begin, ensure you have the following installed:

*   Python 3.7+ 
*   pip (Python package installer) 

### Clone the Repository 

```bash
git clone https://github.com/NotJEEtard-18/ANN-Project-Customer-Churn-Analysis.git
cd ANN-Project-Customer-Churn-Analysis/MultipleFiles
```

### Install Dependencies 

It is recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

**Note:** The `requirements.txt` file should contain all necessary libraries, including `streamlit`, `numpy`, `tensorflow`, `scikit-learn`, and `pandas`. If it's missing, you can create it by running `pip freeze > requirements.txt` after installing the libraries manually, or simply install them one by one:

```bash
pip install streamlit numpy tensorflow scikit-learn pandas
```

## **Usage** 

### Running the Streamlit Application 

To start the web application, navigate to the `MultipleFiles` directory (if you haven't already) and run:

```bash
streamlit run app.py
```

This command will open the Streamlit application in your default web browser. If it doesn't open automatically, Streamlit will provide a local URL (e.g., `http://localhost:8501`) that you can copy and paste into your browser. 🔗

### Making Predictions 🔮

Once the Streamlit application is running:

1.  **Input Customer Details:** Use the interactive widgets (select boxes, sliders, number inputs) to enter the details for the customer you want to predict churn for. ✍️
2.  **View Prediction:** The application will automatically display the "Churn Probability" and a statement indicating whether the customer is likely to churn based on a 0.5 probability threshold. 🤔➡️✅/❌

## **Model Details** 🧠

The `model.h5` file contains the pre-trained Artificial Neural Network. The `label_encoder_gender.pkl`, `onehotencoder.pkl`, and `scalar.pkl` files are pickled objects used for preprocessing the input data consistently with how the model was trained.

*   **`model.h5`**: The saved Keras model (ANN architecture and weights). 
*   **`label_encoder_gender.pkl`**: A `LabelEncoder` object fitted on the 'Gender' column. 
*   **`onehotencoder.pkl`**: A `OneHotEncoder` object fitted on the 'Geography' column. 
*   **`scalar.pkl`**: A `StandardScaler` object fitted on the numerical features. 

These files are crucial for the `app.py` script to load the model and correctly preprocess new input data for prediction. 

## **File Structure** 

```
ANN-Project-Customer-Churn-Analysis/
├── MultipleFiles/
│   ├── app.py                      # Streamlit web application script 
│   ├── model.h5                    # Pre-trained Keras ANN model 
│   ├── label_encoder_gender.pkl    # Pickled LabelEncoder for Gender 
│   ├── onehotencoder.pkl           # Pickled OneHotEncoder for Geography 
│   ├── scalar.pkl                  # Pickled StandardScaler for numerical features 
│   ├── Churn_Modelling.csv         # Dataset used for training (and demonstration) 
│   └── requirements.txt            # Python dependencies 
└── README.md                       # Project README file 
```

## **Contributing** 

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository. 
2.  Create a new branch (`git checkout -b feature/YourFeatureName` or `bugfix/FixBugName`). 
3.  Make your changes. 
4.  Commit your changes (`git commit -m 'Add new feature'`). 
5.  Push to the branch (`git push origin feature/YourFeatureName`). 
6.  Open a Pull Request. 

## **Contact** 

For any questions or inquiries, please contact:

*   **Your Name:** Shubham Kumar Jha(shubham.kr.jha.2005@gmail.com) 
*   **GitHub:** [https://github.com/NotJEEtard-18](https://github.com/NotJEEtard-18) 

---
