from django.urls import path, include
from . import views
from django.contrib.auth import views as auth_views
from .views import create_checkout_session,payment_view,GeneratePaymentReceiptPDF,payment_record_view

app_name = 'user'

urlpatterns = [
    # path('register/',views.register,name='user-register'),
    path('register/', views.RegisterView.as_view(), name='user-register'),

    path('login/',views.MyLoginView.as_view(),name="user-login"),
    path('logout/',views.MyLogoutView.as_view(),name="user-logout"),
    path('profile/',views.ProfileView,name="user-profile"),
    path('editprofile/',views.EditProfileView.as_view(),name="user-editprofile"),
    
    path('user-apartment/<int:apartment_id>/', views.user_apartment_detail, name='user-apartment-detail'),
    
    path('book_success/', views.book_success_view, name='book-success'),
    path('book_fail/', views.book_fail_view, name='book-fail'),
    
    path('rent_success/', views.rent_success_view, name='rent-success'),
    path('rent_fail/', views.rent_fail_view, name='rent-fail'),
    
    

    path('create-checkout-session/<int:apartment_id>/', create_checkout_session, name='create-checkout-session'),
    
    path('success/', views.success_view, name='success'),
    path('cancel/', views.cancel_view, name='cancel'),
    
    path('payment/', payment_view, name='payment'),
    
    

    path('payment_record/', views.payment_record_view, name='payment-record'),
    path('payment_record/payment-receipt/<int:payment_id>/', GeneratePaymentReceiptPDF.as_view(), name='generate_payment_receipt_pdf'),
]

