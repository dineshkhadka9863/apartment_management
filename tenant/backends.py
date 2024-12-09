from django.contrib.auth.backends import ModelBackend
from .models import Tenant

class TenantBackend(ModelBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            tenant = Tenant.objects.get(username=username)
            if tenant.password == password:
                user = tenant.user  # Assuming the Tenant model has a OneToOneField to the User model
                return user
        except Tenant.DoesNotExist:
            pass
        return None
