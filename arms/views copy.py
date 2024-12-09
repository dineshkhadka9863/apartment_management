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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Create your views here.

class ListingView(ListView):
    model = Apartment
    template_name = "dashboard/listing.html"
    context_object_name = "apartment"

    def get_queryset(self):
        # # Get all apartments
        queryset = super().get_queryset()

        # Get the list of apartment IDs from the UserApartment model
        excluded_apartment_ids = UserApartment.objects.values_list(
            "apartment_id", flat=True
        )

        #         # Filter the queryset to include only apartments not in UserApartment model
        queryset = queryset.exclude(id__in=excluded_apartment_ids)

        # Get the search query from the request parameters
        search_query = self.request.GET.get("search")
        location = self.request.GET.get("location")
        floor = self.request.GET.get("floor")
        bhk = self.request.GET.get("bhk")
        max_price = self.request.GET.get("price")

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

        return queryset



#working recommendation
def preprocess_data(apartments):
    vectorizer = cache.get("vectorizer")
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        descriptions = [apartment.description for apartment in apartments]
        description_vectors = vectorizer.fit_transform(descriptions)
        cache.set("vectorizer", vectorizer, timeout=None)  # Cache the vectorizer indefinitely
    else:
        description_vectors = vectorizer.transform(
            [apartment.description for apartment in apartments]
        )
    return vectorizer, description_vectors


def calculate_similarity(user_preferences, vectorizer, description_vectors):
    print("Description vector")
    print(description_vectors)
    user_description = user_preferences.get("description", "")  # Get the description from user preferences
    user_description_vector = vectorizer.transform([user_description])
    similarities = cosine_similarity(user_description_vector, description_vectors)
    return similarities.flatten()


def extract_numeric_bhk(bhk_string):
    numeric_part = bhk_string[:-3]  
    return int(numeric_part)

def map_floor_to_integer(floor_value):
    if floor_value == "Ground":
        return 0
    elif floor_value == "First":
        return 1
    elif floor_value == "Second":
        return 2 
    elif floor_value == "Third":
        return 3
    else:
        return int(floor_value)


def recommend_apartments(user_preferences, apartments, top_n=4):
    vectorizer, description_vectors = preprocess_data(apartments)
    similarities = calculate_similarity(
        user_preferences, vectorizer, description_vectors
    )
    print("Similarities")
    print(similarities)
    apartment_scores = [
        (apartment, score) for apartment, score in zip(apartments, similarities)
    ]
    # Sort the apartments based on similarity score and other factors (location, price, floor, bhk, wifi, parking, swimming pool, ac)
    apartment_scores.sort(
        key=lambda x: (
            x[0].location == user_preferences["location"],
            -extract_numeric_bhk(x[0].bhk),  # Convert to integer and negate
            -map_floor_to_integer(x[0].floor),  # Convert to integer and negate
            -int(x[0].wifi),   # Convert to integer and negate (True -> -1, False -> 0)
            -int(x[0].parking),  # Convert to integer and negate (True -> -1, False -> 0)
            -int(x[0].swimming_pool),  # Convert to integer and negate (True -> -1, False -> 0)
            -int(x[0].ac),  # Convert to integer and negate (True -> -1, False -> 0)
            -int(x[0].price),  # Convert to integer and negate
        ),
        reverse=True,
    )
    top_apartments = [apartment for apartment, _ in apartment_scores][:top_n]
    return top_apartments



def apartment_details(request, apartment_id):
    all_apartment = Apartment.objects.all()
    apartment = get_object_or_404(Apartment, id=apartment_id)

    user_preferences = {
        "description": apartment.description,
        "bhk": apartment.bhk,
        "floor": apartment.floor,
        "parking": apartment.parking,
        "wifi": apartment.wifi,
        "swimming_pool": apartment.swimming_pool,
        "ac": apartment.ac,
        "location": apartment.location,
    }

    apartments = Apartment.objects.exclude(id=apartment_id).prefetch_related()

    recommended_apartments = recommend_apartments(user_preferences, apartments)

    return render(
        request,
        "dashboard/apartment_details.html",
        {
            "apartment": apartment,
            "recommended_apartments": recommended_apartments,
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


class BookView(View):
    def get(self, request, apartment_id=None, username=None):
        if apartment_id and username:
            try:
                apartment = get_object_or_404(Apartment, id=apartment_id)
                user = get_object_or_404(User, username=username)
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
