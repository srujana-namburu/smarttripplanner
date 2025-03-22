import random
from django.db import models
from django.contrib.auth.models import User

class UserOTP(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    otp = models.CharField(max_length=6)
    created_at = models.DateTimeField(auto_now=True)

    def generate_otp(self):
        self.otp = str(random.randint(100000, 999999))
        self.save()
from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pics/', default='default.jpg', blank=True, null=True)
    security_question = models.CharField(max_length=255)
    security_answer = models.CharField(max_length=255)

    def __str__(self):
        return self.user.username
