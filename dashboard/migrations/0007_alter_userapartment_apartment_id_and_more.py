# Generated by Django 4.1 on 2023-06-08 09:28

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tenant', '0009_alter_apartment_price'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('dashboard', '0006_alter_userapartment_apartment_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userapartment',
            name='apartment_id',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='tenant.apartment'),
        ),
        migrations.AlterField(
            model_name='userapartment',
            name='username',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
