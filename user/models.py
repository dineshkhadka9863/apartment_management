from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from tenant.models import Apartment

# Create your models here.
# class User(models.Model):
#     username = models.CharField(max_length=100, unique=True, null=False)
#     email = models.EmailField(unique=True, null=False)
#     password = models.CharField(max_length=100, null=False)
#     first_name = models.CharField(max_length=100, null=False)
#     last_name = models.CharField(max_length=100, null=False)
#     date_of_birth = models.DateField(null=False)
#     gender = models.CharField(max_length=10, null=False)
#     phone_number = models.CharField(max_length=20, null=False, unique=True)
#     address = models.CharField(max_length =250,null=False)
    
class User(AbstractUser):
    phone_number = models.CharField(max_length=20, null=False, unique=True)
    date_of_birth = models.DateField(null=True)
    gender = models.CharField(max_length=10,null=True) 
    first_name = models.CharField(max_length=20,null=False)
    last_name = models.CharField(max_length=20, null= False)
    email = models.EmailField(max_length=30, null=False)
    is_admin = models.BooleanField(default=False)
    is_user = models.BooleanField(default=True)


class Payment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    apartment = models.ForeignKey(Apartment, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Payment #{self.pk}"


