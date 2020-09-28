from flask import Flask, render_template, url_for, Response

# Define app
app = Flask(__name__)


# Create Webpage that displays vitals
# Homepage
@app.route('/')
@app.route('/home')
@app.route('/vitals')
def vitals():
	return render_template('vitals.html')

# Access Camera and Post on Cam box


# # Dashboard
# @app.route('/dashboard')
# def dashboard():
#     return render_template('dashboard.html', title = 'Dashboard')

# # Description
# @app.route('/description')
# def description():
#     return render_template('description.html', title = 'Description')

# About
@app.route('/about')
def about():
    return render_template('about.html', title = 'About')

# # Updates
# @app.route('/updates')
# def updates():
#     return render_template('updates.html', title = 'Updates')

# # Contact
# @app.route('/contact')
# def contact():
#     return render_template('contact.html', title = 'Contact')


if __name__ == '__main__':
	app.run(debug=True)

