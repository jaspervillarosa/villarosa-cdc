from django import forms
from .models import CoconutDamages

class CoconutDamagesForm(forms.ModelForm):
    class Meta:
        model = CoconutDamages
        fields = ['name', 'biologicalcontrol', 'culturalcontrol', 'chemicalcontrol']