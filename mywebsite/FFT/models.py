from django.db import models

# Create your models here.
class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    transformed_image = models.ImageField(upload_to='transformed_images/', blank=True, null=True)  
    created_at = models.DateTimeField(auto_now_add=True)    
  