from flask import Flask, render_template, url_for

# Define app
app = Flask(__name__)


# Create Webpage that displays vitals
@app.route('/')
@app.route('/vitals')
def vitals():
	return render_template('vitals.html')

if __name__ == '__main__':
	app.run(debug=True)

