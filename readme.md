
python -m venv .venv

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process



.venv\Scripts\activate

source .venv/bin/activate   ---------->  ubuntu

python -m w1


pip install -r requirements.txt




pyinstaller --onefile -n w1  w1\__main__.py


pyinstaller --onefile --icon=icon.ico -n m2 w1\__main__.py


pip install --upgrade ultralytics


pip freeze > requirements.txt – შექმნა/განახლება

pip install -r requirements.txt – dependency–ის ინსტალაცია