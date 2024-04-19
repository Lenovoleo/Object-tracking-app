from django.contrib import admin
from django.urls import path
from main.views import video_feed

urlpatterns = [
    path('admin/', admin.site.urls),
    path('video_feed/', video_feed),
]
