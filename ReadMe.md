# Book Page Classifier API

This is a FastAPI application that serves a PyTorch model for classifying book pages. The API allows you to upload an image of a book page, get a prediction for its class, and then update the prediction if it's incorrect.

## Features

-   **Predict Page Class:** Upload an image and get a class prediction (Cover, Title Page, Back Title Page, Colophon).
-   **Store Predictions:** Predictions and images are stored in a PostgreSQL database.
-   **Correct Predictions:** Update the predicted class with a user-defined class.

## Project Structure

.├── app│   ├── init.py│   ├── crud.py│   ├── database.py│   ├── main.py│   ├── model.py│   ├── models.py│   ├── preprocessing.py│   └── schemas.py├── tests│   ├── init.py│   └── test_main.py├── .env├── book_page_classifier.pth└── requirements.txt
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the PostgreSQL database:**
    -   Make sure you have PostgreSQL installed and running.
    -   Create a new database for this application.
    -   Create a `.env` file in the root of the project and add your database URL:
        ```
        DATABASE_URL="postgresql://user:password@postgresserver/db"
        ```

5.  **Place your model file:**
    -   Make sure the `book_page_classifier.pth` file is in the root of the project directory.

## Running the Application

To run the FastAPI application locally, use the following command:

```bash
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000.API EndpointsPOST /predict/Description: Upload an image to get a class prediction.Request Body: multipart/form-data with a file field named file.Response:{
  "id": 1,
  "predicted_class": 0
}
PUT /predict/{image_id}Description: Update the user-defined class for a specific image.Path Parameter: image_id (integer) - The ID of the image to update.Query Parameter: user_class (integer) - The new class for the image.Response:{
  "id": 1,
  "predicted_class": 0,
  "user_class": 1
}
TestingTo run the tests for this application, use pytest:pytest
