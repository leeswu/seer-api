from flask import Flask, render_template, redirect, url_for, request, flash, session
from forms import UploadForm
import tempfile
from gptProcessor import GPTProcessor
import dotenv
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'f1cacf64ffc7cb8983e52ba34cd39b09'
dotenv.load_dotenv()

@app.route("/", methods=["GET"])
def home():
    form = UploadForm()
    return render_template("home.html", form=form)

@app.route("/upload", methods=["POST"])
def upload():
    transcript = None
    
    # set up form for file upload
    form = UploadForm()

    if form.validate_on_submit():
        file = form.file.data
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp_file:
            file.save(tmp_file.name)
            
            gpt = GPTProcessor(os.getenv('OPENAI_API_KEY'))
            gpt.get_pages(tmp_file.name)
            
            alt_text = gpt.get_alt_text()
            
            print(alt_text)
            raw_transcript = gpt.get_raw_transcript()
            transcript = gpt.get_structured_transcript(raw_transcript, alt_text)
            
            # save transcript to file
            with open("static/example-v2.html", "w") as f:
                f.write(transcript) 
                    
        # file is cleaned up upon close
        tmp_file.close()
        
        flash("File uploaded successfully.", "success")
    else:
        flash("Invalid file. Please upload a PDF file.", "error")
    return redirect(url_for('processed', transcript=transcript))

@app.route("/processed", methods=["GET"])
def processed():
    transcript = request.args.get('transcript')
    # Initialize chat history if it doesn't exist
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('processed.html', 
                         transcript=transcript, 
                         chat_history=session['chat_history'])

@app.route("/example-v1", methods=["GET"])
def example_v1():
    return redirect(url_for('static', filename='example-v1.html'))

@app.route("/about")
def about():
    return "<p>About me</p>"

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.form.get('question')
    if not question:
        flash("Please enter a question", "error")
        return redirect(url_for('processed'))

    # Add user question to chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append({
        'role': 'user',
        'content': question
    })

    try:
        # Use GPTProcessor to get response
        gpt = GPTProcessor(os.getenv('OPENAI_API_KEY'))
        response = gpt.ask_about_content(question)
        
        # Add assistant response to chat history
        session['chat_history'].append({
            'role': 'assistant',
            'content': response
        })
        session.modified = True

    except Exception as e:
        flash("Error processing your question. Please try again.", "error")
        print(f"Error: {str(e)}")

    return redirect(url_for('processed'))

if __name__ == '__main__':
    app.run(debug=True, port=8000)