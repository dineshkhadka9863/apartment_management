# Generated by Django 4.1 on 2023-06-07 08:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tenant', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Apartment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('apartment_id', models.IntegerField(unique=True)),
                ('location', models.CharField(max_length=100)),
                ('price', models.DecimalField(decimal_places=2, max_digits=10)),
                ('description', models.TextField()),
                ('image', models.ImageField(default='', upload_to='images')),
            ],
        ),
        migrations.DeleteModel(
            name='Tenant',
        ),
    ]