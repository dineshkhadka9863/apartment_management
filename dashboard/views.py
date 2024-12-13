from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import CreateView, ListView
from django.views import View
from django.urls import reverse_lazy
from tenant.models import Apartment, UserPreferences
from .models import Support, UserApartment, Book, Rent
from .forms import SupportForm
from .models import User, Rent, User, UserApartment, Book
from django.urls import reverse
from django.db.models import Q
from django.db import IntegrityError
from django.core.cache import cache
import numpy as np
from .cosine_similarity import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Create your views here.

class ListingView(ListView):
    model = Apartment
    template_name = "dashboard/listing.html"
    context_object_name = "apartment"

    def get_queryset(self):
        # Get all apartments
        queryset = super().get_queryset()

        # Get the list of apartment IDs from the UserApartment model
        excluded_apartment_ids = UserApartment.objects.values_list(
            "apartment_id", flat=True
        )

       
        queryset = queryset.exclude(id__in=excluded_apartment_ids)

        # Get the search query from the request parameters
        search_query = self.request.GET.get("search")
        location = self.request.GET.get("location")
        floor = self.request.GET.get("floor")
        bhk = self.request.GET.get("bhk")
        max_price = self.request.GET.get("price")
        apartment_id = self.request.GET.get("apartment_id")

        # If a search query is provided, filter the apartments based on the search query
        if search_query:
            queryset = queryset.filter(
                Q(location__icontains=search_query)
                | Q(floor__icontains=search_query)
                | Q(bhk__icontains=search_query)
                | Q(apartment_id__icontains=search_query)
            )

        # Filter by location
        if location:
            queryset = queryset.filter(location=location)

        # Filter by maximum price
        if max_price:
            queryset = queryset.filter(price__lte=max_price)

        if floor:
            queryset = queryset.filter(floor=floor)

        if bhk:
            queryset = queryset.filter(bhk=bhk)

        if apartment_id:
            queryset = queryset.filter(apartment_id=apartment_id)

        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get unique locations from database
        context['locations'] = Apartment.objects.values_list('location', flat=True).distinct()
        return context




class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.enc = OneHotEncoder()
        self.enc.fit(X)
        return self
    
    def transform(self, X):
        return self.enc.transform(X)



def apartment_details(request, apartment_id):
    apartment = get_object_or_404(Apartment, id=apartment_id)
    all_apartments = Apartment.objects.all()
    
    # Filter apartments by same location first, excluding current apartment
    apartments_data = [apt for apt in all_apartments 
                      if apt.location == apartment.location and apt.id != apartment_id]
    
    if not apartments_data:
        return render(
            request,
            "dashboard/apartment_details.html",
            {
                "apartment": apartment,
                "recommended_apartments": [],
                "similarity_scores_dict": {},
            },
        )

    try:
        # Define fixed price range (±100,000)
        price_range = 100000  # Fixed range of 100,000
        min_price = apartment.price - price_range
        max_price = apartment.price + price_range
        
        # Filter apartments within price range
        apartments_in_range = [apt for apt in apartments_data 
                             if min_price <= apt.price <= max_price]
        
        if not apartments_in_range:
            # If no apartments in range, take closest ones by price
            apartments_data.sort(key=lambda x: abs(x.price - apartment.price))
            apartments_in_range = apartments_data[:4]

        # Prepare feature matrix including current apartment
        features = []
        all_apts = [apartment] + apartments_in_range
        
        for apt in all_apts:
            # Convert boolean values to integers
            parking = 1 if apt.parking else 0
            wifi = 1 if apt.wifi else 0
            swimming_pool = 1 if apt.swimming_pool else 0
            ac = 1 if apt.ac else 0
            
            # Extract BHK number
            bhk_num = float(apt.bhk.replace('BHK', '').strip())
            
            # Convert floor to numerical value
            floor_map = {'Ground': 0, 'First': 1, 'Second': 2, 'Third': 3}
            floor_num = floor_map.get(apt.floor, 0)
            
            features.append([
                bhk_num * 2,  # Give more weight to BHK
                floor_num,
                parking,
                wifi,
                swimming_pool,
                ac,
                float(apt.price) / 10000  # Normalize price to smaller scale
            ])

        # Convert to numpy array
        features = np.array(features, dtype=float)
        
        # Normalize features
        max_vals = np.max(features, axis=0)
        min_vals = np.min(features, axis=0)
        features_normalized = (features - min_vals) / (max_vals - min_vals + 1e-10)
        
        # Calculate similarities using custom implementation
        similarities = []
        reference_features = features_normalized[0]
        
        for features in features_normalized[1:]:
            similarity = cosine_similarity(reference_features, features)
            similarities.append(similarity)
        
        # Create (apartment, score) pairs and sort by similarity
        apartment_similarity = list(zip(apartments_in_range, similarities))
        
        # Sort primarily by price proximity to reference apartment, then by similarity
        apartment_similarity.sort(
            key=lambda x: (abs(x[0].price - apartment.price), -x[1])
        )
        
        # Get top 4 recommendations
        recommended_apartments = [apt for apt, _ in apartment_similarity[:4]]
        
        # Create similarity scores dictionary
        similarity_scores_dict = {
            apt.id: score 
            for apt, score in apartment_similarity[:4]
        }
        
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        recommended_apartments = []
        similarity_scores_dict = {}

    return render(
        request,
        "dashboard/apartment_details.html",
        {
            "apartment": apartment,
            "recommended_apartments": recommended_apartments,
            "similarity_scores_dict": similarity_scores_dict,
        },
    )




class SupportView(CreateView):
    form_class = SupportForm
    template_name = "dashboard/complaint.html"
    success_url = reverse_lazy("dashboard:dashboard-listing")


def removeComplaint(request, pk):
    complaint = get_object_or_404(Support, pk=pk)
    complaint.delete()
    return redirect("tenant:tenant-complaint-list")


from django.db import IntegrityError
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.views import View

class BookView(View):
    def get(self, request, apartment_id=None, username=None):
        if apartment_id and username:
            try:
                apartment = get_object_or_404(Apartment, id=apartment_id)
                user = get_object_or_404(User, username=username)
                
                if Book.objects.filter(username=user, apartment_id=apartment).exists():
                    return redirect(reverse("user:book-fail"))
                
                book = Book(username=user, apartment_id=apartment)
                book.save()
                return redirect(reverse("user:book-success"))
            except IntegrityError:
                return redirect(reverse("user:book-fail"))



def removeBooked(request, pk):
    booked = get_object_or_404(Book, pk=pk)
    booked.delete()
    return redirect("tenant:tenant-book")


class RentView(View):
    def get(self, request, apartment_id=None, username=None):
        if apartment_id and username:
            try:
                apartment = get_object_or_404(Apartment, id=apartment_id)
                user = get_object_or_404(User, username=username)
                rent = Rent(username=user, apartment_id=apartment)
                rent.save()
                return redirect(reverse("user:rent-success"))
            except IntegrityError:
                return redirect(reverse("user:rent-fail"))


def removeRented(request, pk):
    rented = get_object_or_404(Rent, pk=pk)
    rented.delete()
    return redirect("tenant:tenant-rent")















# class ListingView(ListView):
#     model = Apartment
#     template_name = 'dashboard/listing.html'
#     context_object_name = "apartment"

#     def get_queryset(self):
#         # Get all apartments
#         queryset = super().get_queryset()

#         # Get the search query from the request parameters
#         search_query = self.request.GET.get('search')

#         # If a search query is provided, filter the apartments based on the search query
#         if search_query:
#             queryset = queryset.filter(
#                 Q(location__icontains=search_query) |
#                 Q(apartment_id__icontains=search_query)
#             )

#         # Get the list of apartment IDs from the UserApartment model
#         excluded_apartment_ids = UserApartment.objects.values_list('apartment_id', flat=True)

#         # Exclude the apartments that have corresponding entries in UserApartment model
#         queryset = queryset.exclude(id__in=excluded_apartment_ids)

#         return queryset










# def apartment_details(request, apartment_id):
#     all_apartment = Apartment.objects.all()
#     apartment = get_object_or_404(Apartment, id=apartment_id)
#     booked_apartments = get_object_or_404(Apartment, id=apartment_id)

#     user_preferences = {
#         "description": apartment.description,
#         "bhk": apartment.bhk,
#         "floor": apartment.floor,
#         "parking": apartment.parking,
#         "wifi": apartment.wifi,
#         "swimming_pool": apartment.swimming_pool,
#         "ac": apartment.ac,
#         "location": apartment.location,
#     }
#     print("user prefences")
#     print(user_preferences)
#     print("user prefences end")

#     apartments = Apartment.objects.exclude(id=apartment_id)

#     recommended_apartments = recommend_apartments(user_preferences, apartments)

#     return render(
#         request,
#         "dashboard/apartment_details.html",
#         {
#             "apartment": apartment,
#             "booked_apartments": booked_apartments,
#             "recommended_apartments": recommended_apartments,
#         },
#     )
