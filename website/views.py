from django.shortcuts import render, HttpResponse
from chemistry.visual_script import upload_csv
# View is where you allow for contents to be seen
# Create your views here.

# below is a func/method , it "renders" the html
def home(request):
    return render(request, "home.html")

upload_csv(
    #user_file_input
)
