# Classifii : Active Learning with Rationales for Text Classification

Authors : Alice Petit, Aurélien Chambon, Canozum Erydn, Jules Carpentier, Yanni Blier

## After you cloned

Download the 20newsgroup dataset and GloVe files in a directory called `data`

In `classifii` directory after activating your virtualenv:
```
pip install -r requirements.txt
```

In `justifii` directory:

(make sure you have nodejs and npm installed)
```
npm install
npm run build
```

## To run the `justifii` app

In `classifii` directory:
```
export FLASK_APP=justifii
export FLASK_ENV=development
flask run
```