from django.contrib import admin
from .models import Book, Rent, Support,UserApartment

# Register your models here.


class BookAdmin(admin.ModelAdmin):    
        list_display=('username','apartment_id')
admin.site.register(Book, BookAdmin)

class RentAdmin(admin.ModelAdmin):
        list_display=('username','apartment_id',)
admin.site.register(Rent, RentAdmin)

class SupportAdmin(admin.ModelAdmin):
        list_display=('name','email','phone_number','apartment_id','message')
admin.site.register(Support, SupportAdmin)



class UserApartmentAdmin(admin.ModelAdmin):        
        list_display=('username','apartment_id')
admin.site.register(UserApartment, UserApartmentAdmin)