from flask import Flask, request, render_template_string, url_for
import subprocess
import os
import shutil

app = Flask(__name__)
app.secret_key = 'replace-with-your-secret-key'

# Path to your CLI script
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'VotingVest.py')  # adjust filename
# Directory to store generated images
IMAGE_DIR = os.path.join(app.static_folder, 'images')

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Technical Indicator Backtester</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem; }
      label { display: block; margin-top: 1rem; }
      input, button { padding: 0.5rem; font-size: 1rem; }
      .output { margin-top: 2rem; white-space: pre-wrap; background: #f5f5f5; padding: 1rem; border-radius: 4px; }
      .images { margin-top: 1rem; }
      .images img { max-width: 100%; height: auto; margin-bottom: 1rem; }
    </style>
  </head>
  <body>
    <h1>Stock Indicator Backtester</h1>
    <form method="post">
      <label>Ticker symbol:<br><input name="ticker" required placeholder="e.g. AAPL"></label>
      <label>Threads:<br><input name="threads" type="number" value="20" min="1"></label>
      <label>Member (odd number):<br><input name="member" type="number" value="7" min="1" step="2"></label>
      <label>Start date (YYYY-MM-DD):<br><input name="start" type="date"></label>
      <label>End date (YYYY-MM-DD):<br><input name="end" type="date"></label>
      <button type="submit">Run Backtest</button>
    </form>

    {% if output %}
      <div class="output">
        <h2>Console Output:</h2>
        {{ output }}
      </div>
    {% endif %}

    {% if images %}
      <div class="images">
        <h2>Generated Charts:</h2>
        {% for img in images %}
          <img src="{{ url_for('static', filename='images/' + img) }}" alt="{{ img }}">
        {% endfor %}
      </div>
    {% endif %}
  </body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    images = []

    if request.method == 'POST':
        # Clean previous images
        shutil.rmtree(IMAGE_DIR)
        os.makedirs(IMAGE_DIR, exist_ok=True)

        # Build command
        ticker = request.form['ticker']
        threads = request.form.get('threads', '20')
        member = request.form.get('member', '7')
        start = request.form.get('start')
        end = request.form.get('end')

        cmd = ['python3', SCRIPT_PATH, ticker, '-t', threads, '-m', member]
        if start:
            cmd += ['-s', start]
        if end:
            cmd += ['-e', end]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
        except subprocess.CalledProcessError as e:
            output = f"Error (code {e.returncode}):\n" + e.stderr

        # Move generated PNGs to static/images
        for fname in os.listdir('.'):
            if fname.lower().endswith('.png'):
                shutil.move(fname, os.path.join(IMAGE_DIR, fname))

        # List moved images
        images = sorted(os.listdir(IMAGE_DIR))

    return render_template_string(HTML_TEMPLATE, output=output, images=images)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3322)
