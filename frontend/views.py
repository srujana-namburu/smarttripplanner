from django.http import HttpResponse,JsonResponse
# Import necessary modules and models
import json
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import *
from .virtual_assistant import generate_trip_itinerary
from django.views.decorators.csrf import csrf_exempt
from langchain_ollama import ChatOllama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from django.shortcuts import render
from django.http import JsonResponse
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from django.shortcuts import render
from django.http import JsonResponse
import base64
from io import BytesIO

# Download NLTK data
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv("tourist_review.csv")  # Ensure correct path

# Fill missing values
df['reviews'] = df['review'].fillna("")

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\W', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.lower().strip()
    return ""

df['cleaned_reviews'] = df['reviews'].apply(clean_text)

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['cleaned_reviews'].apply(analyze_sentiment)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_reviews'])

# Mapping sentiment to numerical values
sentiment_map = {"Positive": 1, "Negative": 0, "Neutral": 2}
y = df['sentiment'].map(sentiment_map)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train best model
best_model = LogisticRegression()
best_model.fit(X_train, y_train)

# Function to analyze reviews for a given location
def analyze_location(location):
    location_df = df[df['location'] == location].copy()
    
    if location_df.empty:
        return {"message": f"No reviews available for {location}."}
    
    reviews = location_df['cleaned_reviews']
    X_location = vectorizer.transform(reviews)
    predicted_sentiments = best_model.predict(X_location)
    print(f"Searching for location: {location}")  # Print the input location
    print(df['location'].unique())  # Print unique locations in the dataset


    sentiment_counts = pd.Series(predicted_sentiments).map({1: "Positive", 0: "Negative", 2: "Neutral"}).value_counts()
    
    # Generate word cloud
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(" ".join(reviews))
    
    # Convert pie chart to base64
    fig, ax = plt.subplots()
    sentiment_counts.plot.pie(autopct='%1.1f%%', colors=['lightgreen', 'red', 'gray'], ax=ax)
    ax.set_ylabel("")
    ax.set_title(f"Sentiment Distribution for {location}")
    
    pie_buffer = BytesIO()
    plt.savefig(pie_buffer, format="png")
    plt.close(fig)
    pie_base64 = base64.b64encode(pie_buffer.getvalue()).decode()

    # Convert word cloud to base64
    wc_buffer = BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {location}")
    plt.savefig(wc_buffer, format="png")
    plt.close()
    wordcloud_base64 = base64.b64encode(wc_buffer.getvalue()).decode()

    return {
        "message": f"Sentiment Analysis for {location}",
        "pie_chart": pie_base64,
        "wordcloud": wordcloud_base64
    }

# View to handle sentiment analysis request
def sentiment_analysis_view(request):
    if request.method == "POST":
        location = request.POST.get("location")
        print(location)
        result = analyze_location(location)
        return JsonResponse(result)
    return render(request, "frontend/sentiment_analysis.html")

import os
from django.shortcuts import render

from django.shortcuts import render

def sentiment_analysis(request):
    pass

df = pd.read_csv('safetytips.csv')

# Ensure no NaN values
df['place'] = df['Place'].fillna('').astype(str)
df['season'] = df['Season'].fillna('').astype(str)
df['tips'] = df['Tips'].fillna('')

# Combine 'place' and 'season' for feature extraction
df['combined'] = df['place'] + ' ' + df['season']

# Train the model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['combined'])
y = df['tips']
model = LogisticRegression()
model.fit(X, y)


def get_safety_tips(place):
    """Fetch safety tips for a given place."""
    seasons = df['season'].unique()
    safety_tips = {}
    for season in seasons:
        input_text = f"{place} {season}"
        input_vector = vectorizer.transform([input_text])
        predicted_tips = model.predict(input_vector)[0]
        safety_tips[season] = predicted_tips
    return safety_tips


def safety_tips_view(request):
    """Render safety tips page based on user input."""
    place = request.GET.get('place', '')  # Get the destination
    safety_tips = get_safety_tips(place) if place else {}

    return render(request, 'frontend/travel_safety.html', {'place': place, 'safety_tips': safety_tips})


def plan_trip(request):
    if request.method == "POST":
        destination = request.POST.get("destination")
        num_days = request.POST.get("num_days")

        if not destination or not num_days:
            return render(request, "virtual_assistant.html", {"error": "Please fill in all fields."})

        # Call AI itinerary function
        itinerary = generate_trip_itinerary(destination, int(num_days))

        # Pass generated itinerary to the new page
        return render(request, "frontend/virtual.html", {"destination": destination, "itinerary": itinerary})

    return render(request, "frontend/virtual_assistant.html")

def virtual(request):
    return render(request, 'frontend/virtual.html')

from .tourist_guide import ConversationalAgent

# Replace this with your actual API key
API_KEY = "sk_ZWKljWDc1RQJXV7oFA4fyelPsTuzSN0PEeINR0V3fpk"
agent = ConversationalAgent(api_key=API_KEY)

def explore_place(request):
    if request.method == 'POST':
        place_name = request.POST.get('place_name', '')

        if place_name:
            response = agent.chat(place_name)
        else:
            response = "No place name provided."

        return render(request, 'frontend/explore.html', {'place_name': place_name, 'response': response})

    return render(request, 'frontend/explore.html', {'place_name': '', 'response': 'Invalid request.'})

def hospital_locator_view(request):
    return render(request, "frontend/hospital_locator.html")

@csrf_exempt
def update_location(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        # Save to user's profile
        if request.user.is_authenticated:
            request.user.profile.latitude = latitude
            request.user.profile.longitude = longitude
            request.user.profile.save()
            return JsonResponse({"message": "Location updated successfully"})
        
        return JsonResponse({"error": "User not authenticated"}, status=403)

    return JsonResponse({"error": "Invalid request"}, status=400)

def analyze_sentiment(request):
    return render(request, "frontend/sentiment_analysis.html")



@login_required
def home(request):
    profile = Profile.objects.get(user=request.user)
    return render(request, 'frontend/home.html', {'profile': profile})

def sentiment_analysis(request):
    return render(request, "frontend/sentiment_analysis.html") 

def virtual_assistant(request):
    return render(request, 'frontend/virtual_assistant.html')

def dynamic_pricing(request):
    return render(request, 'frontend/dynamic_pricing.html')

def travel_safety(request):
    return render(request, 'frontend/travel_safety.html')

def hospital_locator(request):
    return render(request, 'frontend/travel_safety.html')

def tourist_guide(request):
    return render(request, 'frontend/tourist_guide.html')

def get_trip_details(request):
    return render(request, 'frontend/get_trip_details.html')

def safety_tips(request):
    return render(request, 'safety_tips.html')

def contact(request):
    return render(request, 'contact.html')
from django.shortcuts import render, redirect ,get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.core.mail import send_mail
from .models import UserOTP
import random
from django.contrib import messages
from .forms import SignUpForm, OTPForm
from django.http import HttpResponse
from .models import Profile

def signup(request):
    print('Request Method:', request.method)
    print('Signup Request Email:', request.POST.get("email"))
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        security_question = request.POST.get('security_question')
        security_answer = request.POST.get('security_answer')
        profile_picture = request.FILES.get('profile_picture')

        if User.objects.filter(email=email).exists():
            return render(request, 'frontend/signup.html', {'error': 'Email already registered'})

        user = User.objects.create_user(username=username, email=email, password=password)
        user.is_active = False  # Activate after OTP verification
        user.save()

        profile = Profile(user=user, security_question=security_question, security_answer=security_answer)
        if profile_picture:
            profile.profile_picture = profile_picture
        profile.save()

        otp_entry, created = UserOTP.objects.get_or_create(user=user)
        otp_entry.otp = str(random.randint(100000, 999999))
        otp_entry.save()

        send_mail(
            'Your OTP Code',
            f'Your OTP is {otp_entry.otp}. Do not share it with anyone.',
            'nandinikorlakanti@gmail.com',
            [user.email],
            fail_silently=False,
        )

        request.session['otp_user_email'] = email
        return redirect('/frontend/verify-otp/')

    return render(request, 'frontend/signup.html')
def verify_otp(request):
    if request.method == 'GET':
        print('session:',request.session)
        print('session otp:',request.session.get('otp_user_email'))
        # print('session otp:',request.session['otp_user_email'])
        if 'otp_user_email' not in request.session:
            return redirect('/signup')

        email = request.session.get('otp_user_email')
        user = User.objects.get(email=email)
        otp_entry = UserOTP.objects.get(user=user)
        if request.method == 'POST':
            entered_otp = request.POST.get('otp')

            if entered_otp == otp_entry.otp:
                user.is_active = True
                user.save()
                login(request, user)  # Log in user after OTP verification
                del request.session['otp_user_email']  # Clear session
                return redirect('http://127.0.0.1:8000/login/')  # Redirect to dashboard after successful verification
            else:
                return render(request, 'frontend/verify_otp.html', {'error': 'Invalid OTP. Try again.'})

    return render(request, 'frontend/verify_otp.html')

def verifyOtp(request):
    if request.method == 'POST':
        entered_otp = request.POST.get('otp')

        email = request.session.get('otp_user_email')
        user = User.objects.get(email=email)
        otp_entry = UserOTP.objects.get(user=user)
        if entered_otp == otp_entry.otp:
            user.is_active = True
            user.save()
            login(request, user)  # Log in user after OTP verification
            # del request.session['otp_user_email']  # Clear session
            request.session.pop('otp_user_email', None)  # ✅ No error if key is missing
            request.session.modified = True  # ✅ Ensure session update

            return redirect('http://127.0.0.1:8000/login/')  # Redirect to dashboard after successful verification
        else:
            return render(request, 'frontend/verify_otp.html', {'error': 'Invalid OTP. Try again.'})


from django.contrib.auth import logout

def login_view(request):
    print(request.method)
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST['password']
        try:
            user = User.objects.get(email = email)
        except User.DoesNotExist:
            messages.error(request , "User With this email does not exist .\n Try Registering")
            return redirect('frontend/signup.html')
        user = authenticate(request,username=user.username, password=password)
        print(user)

        if user is not None:
            login(request,user)
            messages.success(request,'Login Successfull.')
            return redirect('http://127.0.0.1:8000/login/')
            # otp_entry, _ = UserOTP.objects.get_or_create(user=user)
            # otp_entry.generate_otp()

            # send_mail(
            #     'Your OTP Code',
            #     f'Your OTP is {otp_entry.otp}',
            #     'your-email@gmail.com',
            #     [user.email],
            #     fail_silently=False,
            # )
            # request.session['otp_user'] = user.username
            # return redirect('verify_otp')
        else:
            messages.error(request,"Invalid Credentials. Try again")
    return render(request, 'frontend/home.html')

def logout_view(request):
    logout(request)
    return redirect('frontend/login.html')

def update_profile(request):
    if request.method == 'POST':
        user = request.user
        username = request.POST.get('username')
        email = request.POST.get('email')
        profile_picture = request.FILES.get('profile_picture')

        # Update User model
        user.username = username
        user.email = email
        user.save()

        # Update Profile model
        profile, created = Profile.objects.get_or_create(user=user)
        if profile_picture:
            profile.profile_picture = profile_picture
        profile.save()

        return redirect('home')

    return render(request, 'frontend/update_profile.html')