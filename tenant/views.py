from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseForbidden
from django.urls import reverse_lazy
from tenant.models import Apartment
from . forms import ApartmentForm, ApartmentEditForm
from dashboard.models import UserApartment
from dashboard.models import Book, Rent, UserApartment
from user.models import User
from dashboard.models import Support
from django.views.generic import ListView,CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import UpdateView
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
# Create your views here.




def admin_only(view_func):
    def wrapped_view(request, *args, **kwargs):
        if request.user.is_authenticated and request.user.is_admin:
            return view_func(request, *args, **kwargs)
        else:
            return HttpResponseForbidden("You don't have permission to access this page.")
    return wrapped_view

@login_required 
@admin_only
def tenantView(request):
    return render(request, 'tenant/tenant.html')

class ApartmentListView(ListView):
    model = Apartment
    template_name = 'tenant/apartment_list.html'
    context_object_name = "apartment"

def removeApartment(request,pk):
    apt = get_object_or_404(Apartment, pk=pk)
    apt.delete()
    return redirect('tenant:tenant-apartment-list')

class UploadView(CreateView):
    form_class = ApartmentForm
    template_name ='tenant/upload.html'
    success_url = reverse_lazy('tenant:tenant-home')
    
def BookedView(request):
    books = Book.objects.all()
    return render(request, 'tenant/booked.html', {'books': books})

    
# class BookedView(ListView):
#     model = Book
#     template_name = 'tenant/booked.html'
#     context_object_name = "books"

#     def get_queryset(self):
#         # Get all apartments
#         queryset = super().get_queryset()

#         # Get the list of apartment IDs from the UserApartment model
#         excluded_apartment_ids = UserApartment.objects.values_list('apartment_id', flat=True)

#         # Filter the queryset to include only apartments not in UserApartment model
#         queryset = queryset.exclude(id__in=excluded_apartment_ids)

#         return queryset

def RentedView(request): 
    rents = Rent.objects.all()
    return render(request, 'tenant/rented.html', {'rents': rents})

# def UserApartmentRequestView(request,uname,aid):
#     def get(self, request, apartment_id=None, username=None):
#         if apartment_id and username:
#             apartment = get_object_or_404(Apartment, id=apartment_id)
#             user = get_object_or_404(User, username=username)
#             apt = UserApartment(username=user, apartment_id=apartment)
#             apt.save()
#             return redirect(reverse('tenant:tenant-home'))
def UserApartmentRequestView(request, uname, aid):
    if request.method == 'POST':
        # Get the values from the submitted form
        username = uname
        apartment_id = aid
        
        user = User.objects.get(username=username)
        apartment = Apartment.objects.get(apartment_id=apartment_id)

        # Create a new UserApartment instance and save it
        user_apartment = UserApartment(username=user, apartment_id=apartment)
        user_apartment.save()

        # Update the is_available field to False
        apartment.is_available = False
        apartment.save()
        
        # Remove the entry from the Book model
        book = Book.objects.filter(apartment_id=apartment)
        book.delete()
        # Remove the entry from the Rent model
        rents = Rent.objects.filter(apartment_id=apartment)
        rents.delete()
        
        
        # Redirect to a success page or perform any other desired actions
        return redirect('tenant:tenant-rent')

    # Handle GET requests or render the form again in case of errors
    return render(request, 'tenant/user_apartment_request.html')

def removeUserApartment(request,pk):
    user_apt = get_object_or_404(UserApartment, pk=pk)
    user_apt.delete()
    return redirect('tenant:tenant-user-apartment')

def UserApartmentView(request):
    apts = UserApartment.objects.all()
    return render(request,'tenant/user_apartment.html',{'apts':apts})

def ComplaintListView(request):
    complaints = Support.objects.all()
    return render(request,'tenant/complaint_list.html',{'complaints':complaints})



class EditApartmentView(LoginRequiredMixin, UpdateView):
    model = Apartment
    form_class = ApartmentEditForm
    template_name = "tenant/edit_apartment.html"
    success_url = reverse_lazy("tenant:tenant-apartment-list")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        return context

    def form_valid(self, form):
        return super().form_valid(form)