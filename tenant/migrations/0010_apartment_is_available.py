# Generated by Django 4.1 on 2023-06-08 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tenant', '0009_alter_apartment_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='apartment',
            name='is_available',
            field=models.BooleanField(default=True),
        ),
    ]
