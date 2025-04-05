# /text-to-sql-flask/config.py

import os
import torch

class Config:
    """Base configuration settings."""
    # --- Model Configuration ---
    MODEL_NAME = os.environ.get("MODEL_NAME", "TheBloke/sqlcoder-7B-GPTQ")
    HUGGING_FACE_TOKEN = "<INSERT_YOUR_HUGGING_FACE_TOKEN>" # Use env var ideally

    # --- Generation Parameters ---
    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
    TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.1))

    # --- Hardware Settings ---
    USE_GPU = torch.cuda.is_available()
    DEVICE = "cuda:0" if USE_GPU else "cpu"

    # --- Server Configuration ---
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")

    # --- Hardcoded Database Schema ---
    # Define the schema string directly here, same as in the notebook
    DATABASE_SCHEMA = """
    CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary DECIMAL(10, 2),
    HireDate DATE
);

CREATE TABLE Departments (
    DepartmentID INT PRIMARY KEY,
    DepartmentName VARCHAR(50)
);

-- Example: Add foreign key constraint if applicable (helps model understand joins)
-- ALTER TABLE Employees ADD CONSTRAINT FK_Department FOREIGN KEY (DepartmentID) REFERENCES Departments(DepartmentID);

/*
Tips for defining schema:
- Use standard SQL CREATE TABLE syntax.
- Include all relevant tables for your potential questions.
- Include column names and their data types (VARCHAR, INT, DATE, DECIMAL, etc.).
- Specify PRIMARY KEYs.
- Specify FOREIGN KEYs to show relationships between tables.
- Keep it concise but complete for the scope of your questions.
- Comments (like this one) within the schema string are ignored by the model.
*/
    """

# Instantiate the config
config = Config()

# Optional: Print schema on startup for verification
import logging
logging.info("--- Using Hardcoded Database Schema ---")
logging.info(config.DATABASE_SCHEMA)
logging.info("-------------------------------------")