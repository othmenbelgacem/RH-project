# app.py
from flask import Flask, render_template, request
from DatabaseConnect import initialize_firebase, db

app = Flask(__name__)

# Initialize Firebase (you can skip this line if you've already initialized it in databaseconnect.py)
# initialize_firebase()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_resume', methods=['POST'])
def submit_resume():
    name = request.form.get('name')
    resume_content = request.form.get('resume_content')

    # Save the resume to Firebase Firestore
    resumes_ref = db.collection('resumes')
    resumes_ref.add({'name': name, 'resume_content': resume_content})

    return 'Resume submitted successfully!'

if __name__ == '__main__':
    app.run(debug=True)
