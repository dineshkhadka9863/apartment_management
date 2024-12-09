from django.urls import path
from dashboard.views import removeBooked
from . import views

app_name = "dashboard"

urlpatterns = [
    
    path('complaint/',views.SupportView.as_view(), name='dashboard-complaint'),
    path('complaint/<int:pk>',views.removeComplaint, name='remove-complaint'),
    
    path('',views.ListingView.as_view(), name='dashboard-listing'),

    
    path('apartment/<int:apartment_id>/', views.apartment_details, name='apartment-details'),
    
    path('book/<int:apartment_id>/<str:username>/', views.BookView.as_view(), name='dashboard-book'),
    path('book/<int:pk>',views.removeBooked, name='remove-book'),
    
    path('rent/<int:apartment_id>/<str:username>/', views.RentView.as_view(), name='dashboard-rent'),
    path('rent/<int:pk>',views.removeRented, name='remove-rent'),
]
