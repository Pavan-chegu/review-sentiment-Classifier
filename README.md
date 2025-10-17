Review Sentiment Classifier (Flask)

Simple Flask app that classifies movie reviews as positive or negative using a hybrid CNN+LSTM model and displays the result on the same page.

Requirements

Python 3.6+

Dependencies in requirements.txt

Setup (PowerShell)
# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# optional flask secret
$env:FLASK_SECRET = 'a-secret-for-flash'

# run
python app.py

Notes

Enter a review in the web interface at http://127.0.0.1:5000/ to get sentiment prediction.

Uses a hybrid CNN+LSTM model for better accuracy.

Can be extended for batch predictions or API endpoints
