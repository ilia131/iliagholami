from django.shortcuts import render, redirect , HttpResponse
from django.contrib import messages
# Create your views here.
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login , logout


def login_view(request): 
    if request.method=='POST':
       username= request.POST.get('username')
       pass1= request.POST.get('pass')
       print(username,pass1) 
       user=authenticate(request, username=username, password=pass1)
       if user is not None:
           login(request, user)
           return redirect('home1')
       else:
            return HttpResponse('Username or password is not incorrect')
    return render(request, 'login.html')





def register_view(request):
    if request.method == 'POST':
        uname= request.POST.get('username')
        email= request.POST.get('email')
        pass1= request.POST.get('password1')
        pass2= request.POST.get('password2')
        
        if pass1 != pass2:
           return  HttpResponse("Your Password and Confirm password are not same !!!")
        else:
        
            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
        return redirect('Login')
        
      
        
        
    return render(request, 'signup.html')

def logoutPage(request):
    logout(request)
    return redirect ('Login')






