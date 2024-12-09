from django.db import models
from user.models import User
from tenant.models import Apartment
from django import forms


# Create your models here.
   
class Book(models.Model):
    username = models.ForeignKey(User, on_delete=models.CASCADE)  
    apartment_id = models.ForeignKey(Apartment, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.username.username}'s booking"

    
    
class Rent(models.Model):
    username = models.OneToOneField(User, on_delete=models.CASCADE)
    apartment_id = models.ForeignKey(Apartment, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.username.username}'s renting"
    

class UserApartment(models.Model):
    username = models.ForeignKey(User, on_delete=models.CASCADE)  
    apartment_id = models.ForeignKey(Apartment, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.username.username}'s apartment"
    
# class Complaint(models.Model):
#     name = models.CharField(max_length=100, null=False)
#     email = models.EmailField(max_length = 30)
#     phone_number = models.CharField(max_length=20, null=False)
#     apartment_id = models.CharField(null=False, max_length=10)
#     message = models.TextField()

class Support(models.Model):
    name = models.CharField(max_length=100, null=False)
    email = models.EmailField(max_length = 30)
    phone_number = models.CharField(max_length=20, null=False)
    apartment_id = models.CharField(null=False, max_length=10)
    message = models.TextField()
    