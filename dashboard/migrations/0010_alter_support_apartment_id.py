# Generated by Django 4.1 on 2023-06-14 03:54

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tenant', '0011_alter_apartment_location'),
        ('dashboard', '0009_alter_userapartment_apartment_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='support',
            name='apartment_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='tenant.apartment'),
        ),
    ]