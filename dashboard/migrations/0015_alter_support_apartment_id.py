# Generated by Django 4.1 on 2023-06-14 06:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0014_alter_support_apartment_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='support',
            name='apartment_id',
            field=models.CharField(max_length=10),
        ),
    ]
