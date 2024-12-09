from django import forms
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from .models import Support
from tenant.models import Apartment
import re

# class ComplaintForm(forms.ModelForm):
#     class Meta:
#         model = Complaint
#         fields = ('name','email','phone_number','apartment_id','message')
#         widgets ={
#             'name': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Name'}),
#             'email': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Email address'}),
#             'phone_number': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Phone number'}),
#             'apartment_id': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Apartment'}),
#             'message': forms.Textarea(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm','placeholder':'Write your message...'}),
#         }

def validate_phone_number(value):
    if not value.isdigit():
        raise ValidationError(_('Invalid phone number. Please enter only digits.'))
    if len(value) != 10:
        raise ValidationError(_('Invalid phone number. The phone number must have 10 digits.'))

def validate_name(value):
    if not re.match(r'^[a-zA-Z ]+$', value):
        raise ValidationError(_('Invalid characters in the name.'))

class SupportForm(forms.ModelForm):
    
    name = forms.CharField(widget=forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Name'}),validators= [validate_name])
    phone_number = forms.CharField(widget=forms.TextInput(attrs={'class': 'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Phone number'}),validators=[validate_phone_number])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Query available apartments from the database
        available_apartments = Apartment.objects.all().values_list('apartment_id', flat=True)
    
        # Create a list of tuples for choices (apartment_id, apartment_id)
        apartment_choices = [(apartment_id, apartment_id) for apartment_id in available_apartments]

        # Update the choices for the 'apartment_id' field
        self.fields['apartment_id'].widget.choices = apartment_choices
    
    class Meta:
        model = Support
        fields = ('name','email','phone_number','apartment_id','message')
        widgets ={
            # 'name': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Name'}),
            'email': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Email address'}),
            # 'phone_number': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Phone number'}),
            # 'apartment_id': forms.TextInput(attrs={'class':'w-full rounded-lg border-gray-200 p-3 text-sm','placeholder':'Apartment'}),
            'apartment_id': forms.Select(attrs={'class': 'w-full rounded-lg border-gray-200 p-3 text-sm', 'placeholder': 'Apartment'}),
            'message': forms.Textarea(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm','placeholder':'Write your message...'}),
        }        

