from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import UserCreationForm,UserChangeForm
from django.forms.widgets import RadioSelect, SelectDateWidget
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _
from .models import User
from django.contrib.auth import get_user_model
import re


def validate_password(value):
    # Check password length
    if len(value) < 8:
        raise ValidationError("Password must be 8 character which contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")
    # Check password complexity
    if not any(char.isdigit() for char in value) or not any(char.isupper() for char in value) or not any(char.islower() for char in value) or not any(char in '!@#$%^&*()_+.,/?><~`' for char in value):
        raise ValidationError("Password must be 8 character which contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")


def validate_phone_number(value):
    if not value.isdigit():
        raise ValidationError(_('Invalid phone number. Please enter only digits.'))
    if len(value) != 10:
        raise ValidationError(_('Invalid phone number. The phone number must have 10 digits.'))

def validate_name(value):
    if not re.match(r'^[a-zA-Z]+$', value):
        raise ValidationError(_('Invalid characters in the name.'))
    
class UserForm(UserCreationForm):
    
    
    
    # def save(self, commit=True):
    #     user = super().save(commit=False)
    #     user.set_password(self.cleaned_data['password'])
        
    #     confirm_password = self.cleaned_data.get('confirm_password')
    #     if confirm_password:
    #         user.set_password(confirm_password)
        
    #     if commit:
    #         user.save()
    #     return user


    gender_choices = (
        ('male','Male'),('female','Female'),('other','Other'),
        
    )
    
    gender = forms.ChoiceField(choices=gender_choices, widget = RadioSelect)
    date_of_birth = forms.DateField(widget=SelectDateWidget(years=range(1923, 2023)))
    # confirm_password = forms.CharField(label='Confirm Password',widget=forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}))
    # gender = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.RadioSelect(attrs={'class': 'radio'}))
    first_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}), validators=[validate_name])
    last_name = forms.CharField(widget=forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}), validators=[validate_name])
    phone_number = forms.CharField(widget=forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),validators=[validate_phone_number])
    password1 = forms.CharField(label = "Password", widget=forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm custom-password-field'}), validators=[validate_password])
    password2 = forms.CharField(label = "Confirm Password",widget=forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm custom-password-field'}))
    class Meta:
        model = User

        fields = [
            "first_name",
            "last_name",
            "username",
            "email",
            "gender",
            "date_of_birth",
            "phone_number",
            "password1",
            "password2",
        ]
        widgets = {
            'username': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            'email': forms.EmailInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            # 'first_name': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            # 'last_name': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            'date_of_birth': SelectDateWidget(years=range(1923, 2023), attrs={'class': 'form-select'}),
            # 'phone_number': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            'address': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            # 'password1': forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm custom-password-field'}),
            # 'password2': forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            
        }
    
    def clean_password1(self):
        password = self.cleaned_data.get('password1')
        return password
    


class LoginForm(AuthenticationForm):
    
    class Meta:
        model = User

        fields = [
            "username",
            "password",
        ]
        widgets = {
            'username': forms.TextInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm'}),
            'password': forms.PasswordInput(attrs={'class': 'mt-1 w-full rounded-md border-gray-200 bg-white text-sm text-gray-700 shadow-sm custom-password-field'}),
        }
    
# class LoginForm(AuthenticationForm):
   
   
#     # username = forms.CharField(max_length=254,widget=forms.TextInput())
#     # password = forms.CharField(max_length=254,widget=forms.PasswordInput())
    
    
#     def __init__(self, *args, **kwargs):
#         super(LoginForm, self).__init__(*args, **kwargs)
        
#         self.fields['username'].widget.attrs.update({
#             'class': 'w-full rounded-lg border-gray-200 p-4 pe-12 text-sm shadow-sm',
#         })
#         self.fields['password'].widget.attrs.update({
#             'class': 'w-full rounded-lg border-gray-200 p-4 pe-12 text-sm shadow-sm',
#         })
        

class EditProfileForm(UserChangeForm):
    class Meta:
        model = get_user_model()
        fields = (
            
            "first_name",
            "last_name",
            "email",
            "phone_number",

        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove password field
        self.fields.pop("password")
        # Add custom classes to form fields
        for field_name, field in self.fields.items():
            field.widget.attrs.update(
                {
                    "class": "appearance-none block w-full py-2 px-3 leading-tight border border-gray-300 rounded focus:outline-none focus:bg-white focus:border-gray-500"
                }
            )