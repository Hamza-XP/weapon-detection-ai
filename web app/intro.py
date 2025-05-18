from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/detect')
def detect():
    return "Upload page will go here."

if __name__ == '__main__':
    app.run(debug=True)
