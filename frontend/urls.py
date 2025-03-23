from django.contrib import admin
from django.urls import path,include
from frontend import views
from .views import plan_trip


urlpatterns = [
    path('',views.home,name='home'),
    path('home/',views.home,name='home'),
    path('signup/', views.signup, name='signup'),
    path("verify-otp/",views.verify_otp,name = 'verify_otp'),
    path("verifyOtp",views.verifyOtp,name = 'verifyOtp'),
    path('login/', views.login_view, name='login'),
    path('logout',views.logout_view,name='logout'),
    path('home/', views.home, name='home'),
    path('virtual-assistant/', views.virtual_assistant, name='virtual_assistant'),
    path('dynamic-pricing/', views.dynamic_pricing, name='dynamic_pricing'),
    path('travel_safety/', views.travel_safety, name='travel_safety'),
    path('hospital-locator/', views.hospital_locator_view, name='hospital_locator'),
    path('tourist_guide/', views.tourist_guide, name='touristguide'),
    path('get-trip-details/', views.get_trip_details, name='gettripdetails'),
    path('safety_tips/', views.safety_tips_view, name='safety_tips'),
    path('contact/', views.contact, name='contact'),
    path('plan_trip/', views.plan_trip, name='plan_trip'),
    path('virtual/', views.virtual, name='virtual'),
    path('explore_place/', views.explore_place, name='explore_place'),
    path('update-profile/', views.update_profile, name='update_profile'),
    path('logout/', views.logout_view, name='logout'),
    path('update_location/', views.update_location, name='update_location'),
    path('sentiment_analysis/', views.sentiment_analysis, name="sentiment_analysis"),
    
    # Add other paths as needed
]
