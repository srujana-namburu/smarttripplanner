# Generated by Django 5.1.6 on 2025-03-10 18:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('frontend', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='profile',
            name='profile_picture',
            field=models.ImageField(default='default.jpg', upload_to='profile_pics/'),
        ),
    ]
