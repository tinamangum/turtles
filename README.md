# Kenyan Turtle Rescue Project

A data science challenge from Zindi to optimize turtle conservation off of the Kenyan coast. 

Our team delivered a command-line program to predict the number of turtle captures & suggest staff allocation per site -- for the next full calendar year. By conducting a thorough Exploratory Data Analysis of the dataset provided, and by investigating various techniques of machine learning models and algorithms, we improved our baseline model. Our deliverable to the conservation team includes not only a python command-line program, but a set of clear, concrete data requests to vastly improve our modeling. 

Original challenge can be found here: <a href="https://zindi.africa/competitions/sea-turtle-rescue-forecast-challenge">Zindi</a>


________________________________________________________________________
## Set up your Environment



### **`macOS`** type the following commands : 



- For installing the virtual environment and the required package you can either follow the commands:

    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
Or ....
-  use the [Makefile](Makefile) and run `make setup` or install it manually with the following commands:

     ```BASH
    make setup
    ```
    After that active your environment by following commands:
    ```BASH
    source .venv/bin/activate
    ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.11.3
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-bash` CLI :
  
    ```BASH
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/Scripts/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:
    ```Bash
    python.exe -m pip install --upgrade pip
    ```


