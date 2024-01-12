from django.shortcuts import render, HttpResponse
#from chemistry.visual_script import upload_csv
# View is where you allow for contents to be seen
# Create your views here.
#from apply_predictive_model import backendUser
# below is a func/method , it "renders" the html
def home(request):
    return render(request, "home.html")

def step2(request):
    return render(request, "step2.html")

def step1(request):
    return render(request, "step1.html")

def step3(request):
    return render(request, "step3.html")

#def process_csvFile(request):
#    input = request.POST.get("csvFile")
#    output = process_

# upload_csv(
#     #user_file_input
# )
