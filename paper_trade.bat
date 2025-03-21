@echo off
REM Change directory to your project directory
cd /d G:\Repos\Stock-Advizir\

REM Activate the Python environment
REM Replace 'your_env_name' with your actual environment name
call G:\Repos\Stock-Advizir\stckadv_env\Scripts\activate

REM Run the Python script
python G:\Repos\Stock-Advizir\src\paper_test.py

REM Deactivate the Python environment
deactivate