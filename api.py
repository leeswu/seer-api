from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify
from forms import UploadForm
import io
import tempfile
from gptProcessor import GPTProcessor
import dotenv
import os
import time
from flask_cors import CORS

os.makedirs("processed/md", exist_ok=True)
os.makedirs("processed/html", exist_ok=True)

app = Flask(__name__)
CORS(app)
# CSRF protection token
app.config['SECRET_KEY'] = 'f1cacf64ffc7cb8983e52ba34cd39b09'
app.config['PORT'] = 8000

dotenv.load_dotenv()

# Update the streamlit-upload route
@app.route("/streamlit-upload", methods=["POST"])
def streamlit_upload():
    print("upload route hit")
    MD_DIR = "processed/md"
    if "file" not in request.files:
        return jsonify({"error": "No file field in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # make tmp file on disk to pass to PyMuPDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        print("beginning processing")
        # process the temporary file path using GPTProcessor
        gpt = GPTProcessor(os.getenv('OPENAI_API_KEY'))

        # process the file
        pages = gpt.get_pages(tmp_file_path)

        # structure the transcription into a HTML document
        md_doc = gpt.get_structured_md_incremental(pages)

        html_doc = gpt.convert_md_to_html(md_doc)

    except Exception as e:
        print(f"Error converting document: {e}")
        return jsonify({"error": str(e)})

    return jsonify({
        "md": md_doc, 
        "html": html_doc, 
        "filename": file.filename,
        "status": "complete"  # Add status for frontend to know when to play completion sound
    })

@app.route("/", methods=["GET"])
def home():
    form = UploadForm()
    return render_template("home.html", form=form)

# Handle the upload and conversion of the uploaded PDF file
@app.route("/upload", methods=["POST"])
def upload():
    HTML_DIR = "processed/html"
    MD_DIR = "processed/md"

    html_doc = None
    form = UploadForm()

    if form.validate_on_submit():
        file = form.file.data

        # store file into memory (instead of locally)
        file_in_memory = io.BytesIO(file.read())

        # make tmp file on disk to pass to PyMuPDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(file_in_memory.read())
            tmp_file_path = tmp_file.name

        # process the temporary file path using GPTProcessor
        gpt = GPTProcessor(os.getenv('OPENAI_API_KEY'))

        # process the file
        pages = gpt.get_pages(tmp_file_path)

        # get alt text for each page
        alt_text = gpt.get_alt_text(pages)

        # get raw transcription for each page
        raw_transcription = gpt.get_raw_transcription(pages)

        # structure the transcription into a HTML document
        html_doc = gpt.get_structured_html(raw_transcription, alt_text)

        try:
            # save md doc to file
            if raw_transcription:
                MD_FILENAME = f"{form.file.data.filename}-{int(time.time())}.md"
                with open(os.path.join(MD_DIR, MD_FILENAME), "w", encoding="utf-8") as f:
                    f.write(raw_transcription)
        except Exception as e:
            print(f"Error saving MD file: {e}")

        try:
            # save HTML doc to file
            if html_doc:  # Add this check
                HTML_FILENAME = f"{form.file.data.filename}-{int(time.time())}.html"
                with open(os.path.join(HTML_DIR, HTML_FILENAME), "w", encoding="utf-8") as f:
                    f.write(html_doc)
        except Exception as e:
            print(f"Error saving HTML file: {e}")

        flash("File uploaded successfully.", "success")
    else:
        flash("Invalid file. Please upload a PDF file.", "error")

    return redirect(url_for('processed', html_doc=html_doc))

# Display the converted HTML document


@app.route("/processed", methods=["GET"])
def processed():
    html_doc = request.args.get('html_doc')
    if html_doc is None:
        return "No HTML generated."
    return html_doc


@app.route("/example", methods=["GET"])
def example():
    return redirect(url_for('static', filename='/viscomm-short.pdf-1740793822.html'))


if __name__ == '__main__':
    app.run(debug=True, port=app.config['PORT'])
