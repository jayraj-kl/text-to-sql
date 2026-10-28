"""Example database schema used as model context."""

# Replace with the CREATE TABLE statements for your own database.
# The model only knows about the tables and columns listed here.
EXAMPLE_SCHEMA = """
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

# The API has always sent this schema with an indented first line and trailing
# indentation (it used to live inside an indented class body). Whitespace is part
# of the prompt, so it is reproduced byte-for-byte to keep generated SQL unchanged.
API_SCHEMA = "\n    " + EXAMPLE_SCHEMA[1:] + "    "
