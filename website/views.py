from django.shortcuts import render, HttpResponse
# View is where you allow for contents to be seen
# Create your views here.

# below is a func/method , it "renders" the html
def home(request):
    return render(request, "home.html")