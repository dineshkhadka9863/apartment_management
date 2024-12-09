from django.db import models
from django.conf import settings
# Create your models here.

class Apartment(models.Model):
    
    LOCATION_CHOICES = [
        ('Gaushala', 'Gaushala'),
        ('Baneshwor', 'Baneshwor'),
        ('Maharajgunj', 'Maharajgunj'),
    ]
    BHK_CHOICES = [
        ('1BHK','1BHK'),
        ('2BHK','2BHK'),
        ('3BHK','3BHK'),
    ]
    FLOOR_CHOICES =[
        ('Ground','Ground'),
        ('First','First'),
        ('Second','Second'),
        ('Third','Third'),
    ]
    apartment_id = models.IntegerField(null=False, unique=True)
    location = models.CharField(max_length=100, choices=LOCATION_CHOICES)
    price = models.PositiveIntegerField()
    description = models.TextField()
    image = models.ImageField(upload_to='images', default="")
    
    bhk = models.CharField(max_length=100, choices=BHK_CHOICES)
    floor = models.CharField(max_length=100, choices=FLOOR_CHOICES)
    parking = models.BooleanField(default=False)
    wifi = models.BooleanField(default=False)
    swimming_pool = models.BooleanField(default=False)
    ac = models.BooleanField(default=False)
    
    def __str__(self):
        return str(self.apartment_id)


class UserPreferences(models.Model):
    
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    location = models.CharField(max_length=100)
    price = models.PositiveIntegerField()
    bhk = models.CharField(max_length=100)
    floor = models.CharField(max_length=100)
    parking = models.BooleanField(default=False)
    wifi = models.BooleanField(default=False)
    swimming_pool = models.BooleanField(default=False)
    ac = models.BooleanField(default=False)