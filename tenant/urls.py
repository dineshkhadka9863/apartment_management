from django.urls import path
from tenant.views import ComplaintListView
from . import views

app_name = "tenant"

urlpatterns = [
    path('tenant/', views.tenantView, name='tenant-home'),
    path('upload/',views.UploadView.as_view(), name='tenant-upload'),
    path('booked/', views.BookedView, name='tenant-book'),
    path('rented/', views.RentedView, name='tenant-rent'),
    path('complaint_list/',views.ComplaintListView,name='tenant-complaint-list'),
    
    path('accept_apartment/<str:uname>/<int:aid>/', views.UserApartmentRequestView, name='tenant-accept-apartment'),
    
    path('apartment_list/',views.ApartmentListView.as_view(),name='tenant-apartment-list'),
    path('aparmanent_list/<int:pk>',views.removeApartment, name='remove-apartment'),
    
    path('user_apartment/',views.UserApartmentView,name='tenant-user-apartment'),
    path('user_apartment/<int:pk>',views.removeUserApartment, name='remove-user-apartment'),
    
    path('edit_apartment/<int:pk>',views.EditApartmentView.as_view(),name='tenant-edit-apartment'),
]   

