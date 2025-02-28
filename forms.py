from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from flask_wtf.file import FileRequired, FileAllowed

class UploadForm(FlaskForm):
    file = FileField('Upload a PDF file:', validators=[FileRequired(), FileAllowed(['pdf'], 'PDFs only!')])
    submit = SubmitField('Upload')