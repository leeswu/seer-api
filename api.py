from flask import Flask, render_template, redirect, url_for, request, flash, session
from forms import UploadForm
import tempfile
from gptProcessor import GPTProcessor
import dotenv
import os
import time

app = Flask(__name__)

# CSRF protection token
app.config['SECRET_KEY'] = 'f1cacf64ffc7cb8983e52ba34cd39b09'
app.config['PORT'] = 8000

dotenv.load_dotenv()


@app.route("/", methods=["GET"])
def home():
    form = UploadForm()
    return render_template("home.html", form=form)

# Handle the upload and conversion of the uploaded PDF file
@app.route("/upload", methods=["POST"])
def upload():
    # TODO: programatically set up directories
    OUT_DIR = "processed/html"
    html_doc = None
    form = UploadForm()


    if form.validate_on_submit():
        file = form.file.data
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            
            gpt = GPTProcessor(os.getenv('OPENAI_API_KEY'))

            # Get each page as a base64 encoded image
            pages = gpt.get_pages(tmp_file.name)

            # Get alt text for each page    
            alt_text = gpt.get_alt_text(pages)

            # Get raw transcription for each page
            raw_transcription = gpt.get_raw_transcription(pages)

            # Structure the transcription into a HTML document
            html_doc = gpt.get_structured_transcription(raw_transcription, alt_text)

            # save html doc to file
            OUT_FILENAME = f"{form.file.data.filename}-{int(time.time())}.html"
            with open(os.path.join(OUT_DIR, OUT_FILENAME), "w") as f:
                f.write(html_doc) 
                    
        # file is cleaned up upon close
        tmp_file.close()
        
        flash("File uploaded successfully.", "success")
    else:
        flash("Invalid file. Please upload a PDF file.", "error")
    return redirect(url_for('processed', html_doc=html_doc))

# Display the converted HTML document
@app.route("/processed", methods=["GET"])
def processed():
    # TODO: Incorporate the html doc into the processed.html template
    html_doc = request.args.get('html_doc')
    if html_doc is None:
        return "No HTML generated."
    return html_doc


@app.route("/example", methods=["GET"])
def example():
    return redirect(url_for('static', filename='/viscomm-short.pdf-1740793822.html'))


if __name__ == '__main__':
    app.run(debug=True, port=app.config['PORT'])