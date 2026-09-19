@echo off

echo Creating virtual environment...
py -m venv .venv

echo Installing dependencies...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Tetr.AI setup complete!
echo Activate environment with:
echo .venv/Scripts/activate.bat

pause