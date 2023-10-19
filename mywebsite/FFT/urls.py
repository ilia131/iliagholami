from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import  home ,  process_transform
urlpatterns = [
    path('', home, name='home1'),
    path('process_transform/', process_transform, name='process_transform'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)