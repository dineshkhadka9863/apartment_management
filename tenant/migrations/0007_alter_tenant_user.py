# Generated by Django 4.1 on 2023-06-07 14:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('tenant', '0006_alter_tenant_user'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tenant',
            name='user',
            field=models.OneToOneField(default=0, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
