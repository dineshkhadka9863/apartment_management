from  django.shortcuts import render,get_object_or_404,redirect
from django.contrib.auth.forms import AuthenticationForm
from django.views.generic import CreateView,UpdateView
from dashboard.views import apartment_details
from . forms import UserForm, LoginForm,EditProfileForm
from dashboard.models import UserApartment
from django.http import HttpResponseRedirect, HttpResponse, FileResponse
from django.urls import reverse,reverse_lazy
from django.contrib.auth.views import LoginView ,LogoutView
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from user.models import User,Payment
from tenant.models import Apartment
from dashboard.models import Book, Rent
from django.conf import settings
from datetime import datetime
import stripe 
from django.template.loader import get_template
from django.views.generic.base import View
from reportlab.pdfgen import canvas

# Create your views here.
# def register(request):
#     if request.method == 'POST':
#         form = UserForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('user:user-login')
#     else:
#         form = UserForm()

#     context = {
#         'form': form
#     }
#     return render(request, 'user/register.html', context)

# def login(request):
#     form =AuthenticationForm()
#     context ={
#         'form' : form
#     }
#     return render(request,'user/login.html',context)

class RegisterView(CreateView):
    template_name = 'user/register.html'
    form_class = UserForm
    success_url = reverse_lazy('user:user-login')
    
# class MyLoginView(LoginView):
#     form_class = LoginForm
#     template_name = 'user/login.html'

class MyLoginView(LoginView):
    template_name = "user/login.html"
    form_class = LoginForm
    def form_valid(self, form):
        # Log the user in using Django's built-in authentication
        response = super().form_valid(form)

        # Redirect the user based on their role (admin or user)
        user = self.request.user
        if user.is_superuser:
            return redirect('tenant:tenant-home')
        
        elif user.is_user:
            return redirect('dashboard:dashboard-listing')
            
        else:
            return redirect('dashboard:dashboard-listing')

        return response

    def get_success_url(self):
        # Override this method to prevent any further redirection by the LoginView
        return self.request.path
class MyLogoutView(LogoutView):
    next_page = reverse_lazy("user:user-login")
    
def ProfileView(request):
    try:
        user_apartment = UserApartment.objects.get(username=request.user)
    except UserApartment.DoesNotExist:
        user_apartment = None
    
    try:
        book = Book.objects.get(username=request.user)
    except Book.DoesNotExist:
        book = None
    
    try:
        rent = Rent.objects.get(username=request.user)
    except Rent.DoesNotExist:
        rent= None
    
    user = request.user
    payments = Payment.objects.filter(user=user)

    context = {
        'user': request.user,
        'apartment': user_apartment.apartment_id if user_apartment else None,
        'book' : book.apartment_id if book else None,
        'rent' : rent.apartment_id if rent else None,
        'payments': payments
    }
    return render(request, 'user/profile.html', context)
    
class EditProfileView(LoginRequiredMixin, UpdateView):
    model = get_user_model()
    form_class = EditProfileForm
    template_name = "user/editprofile.html"
    success_url = reverse_lazy("user:user-profile")

   
    
    def get_object(self):
        return self.request.user

    def get_context_data(self, **kwargs):
        context = super(EditProfileView, self).get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        return super().form_valid(form)


def user_apartment_detail(request, apartment_id):
    apartment = get_object_or_404(Apartment, id=apartment_id)
    return render(request, 'user/user_apartment_detail.html', {'apartment': apartment})


def book_success_view(request):
    return render(request, 'action/book_success.html')
def book_fail_view(request):
    return render(request, 'action/book_fail.html')

def rent_success_view(request):
    return render(request, 'action/rent_success.html')
def rent_fail_view(request):
    return render(request, 'action/rent_fail.html')



def create_checkout_session(request, apartment_id):
    stripe.api_key = 'sk_test_51NJzcuAps0nGDChLPRJRVOTmAWbsNTmX2zVDjxPD4FjKO13GCfcwQdZLLUnNl9X8X8NBLgCRGnVn3EHhnELvOPv900DlkmZV8X'
    user = request.user
    apartment = Apartment.objects.get(id=apartment_id)


    # Set the apartment_id in the session
    request.session['apartment_id'] = apartment_id

    session = stripe.checkout.Session.create(
        payment_method_types=['card'],
        line_items=[
            {
                'price_data': {
                    'currency': 'npr',
                    'product_data': {
                        'name': apartment.apartment_id,
                    },
                    'unit_amount': min(int(apartment.price * 100),99999999)
                },
                'quantity': 1,
            },
        ],
        mode='payment',
        success_url=request.build_absolute_uri('/success'),
        cancel_url=request.build_absolute_uri('/cancel'),
    )
    return redirect(session.url, code=303)


def success_view(request):
    user = request.user
    apartment_id = request.session.get('apartment_id')
    apartment = Apartment.objects.get(id=apartment_id)
    
    # Save the payment
    payment = Payment(user=user, apartment=apartment, amount=apartment.price, timestamp=datetime.now())
    payment.save()

    # Clear the apartment_id from the session
    del request.session['apartment_id']

    return render(request, 'user/success.html')



def cancel_view(request):
    return render(request, 'user/cancel.html')

def payment_view(request):
    payments = Payment.objects.all()
    return render(request, 'tenant/payment.html', {'payments': payments})


def payment_record_view(request):
    user =request.user
    payments = Payment.objects.filter(user=user)

    context = {
        'user': request.user,
        'payments': payments
    }
    return render(request, 'user/payment_records.html',context)


class GeneratePaymentReceiptPDF(View):
    def get(self, request, payment_id):
        # Retrieve the payment details from the database
        user = request.user
        try:
            payment = Payment.objects.filter(user=user, id=payment_id).latest('timestamp')
        except Payment.DoesNotExist:
            # Handle the case when the payment with the given payment_id does not exist
            return HttpResponse("Payment not found", status=404)

        user = request.user
        payment = Payment.objects.filter(user=user).latest('timestamp')

        #format amount
        formatted_amount = "{:,}".format(int(payment.amount))
        
        # Generate the PDF using ReportLab
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="payment_receipt.pdf"'

        p = canvas.Canvas(response)

        # Customize the receipt layout
        p.setFont("Helvetica", 12)
        
        p.drawString(250, 750, "Payment Receipt")
        p.drawString(100, 700, f"User: {user.username}")
        p.drawString(100, 650, f"Aparment: {payment.apartment.apartment_id}")
        p.drawString(100, 600, f"Amount: Rs. {formatted_amount}")
        p.drawString(100, 550, f"Timestamp: {payment.timestamp}")

        p.showPage()
        p.save()

        return response

