from django.shortcuts import render, HttpResponse
# View is where you allow for contents to be seen
# Create your views here.
def home(request):
    return render(request, "home.html")