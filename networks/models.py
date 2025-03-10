from django.db import models

# Create your models here.
from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()

class UploadFilesForm2(forms.Form):
    file1 = forms.FileField(label='File 1')
    file2 = forms.FileField(label='File 2')
    
