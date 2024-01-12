from django.shortcuts import render, HttpResponse
from chemistry.jupyterbackend import process_csv,make_prediction #image
from chemistry.backendUser import process_molecular_data #describe
# View is where you allow for contents to be seen
# Create your views here.
#from apply_predictive_model import backendUser
# below is a func/method , it "renders" the html
def home(request):
    return render(request, "home.html")


#uploaded_data = upload_csv('file_path')
#then
#parsed_data = parse_data(uploaded_data) 
#visualize_molecules = visualize_molecules(parsed_data) #this is a picture file


#def step1(request):
#    return render(request, "website/step1.html")



def makePredict(request):
    image = make_prediction()
    return HttpResponse(image, content_type="image/svg+xml")

def step1(request):
    return render(request, "step1.html")
def step2(request):
    return render(request, "step2.html")
def step3(request):
    return render(request, "step3.html")

def process_csvFile(request):
    print("starting process")
    input = request.POST.get("csvFile")
    image, details = process_csv(input)
    print("processing step2 request")
    print(details)
    return render(request,"step2.html",{"image":image, "details":details})

def select_algo(request):
    input = request.POST.get(choicealgo) #this is the algo value : describe, visulaize...
    if input == "describe":
        output = process_molecular_data(request, test_size=0.2, random_state=42)
    return render(request, "step3.html",{"algo result":result})
# upload_csv(
#     #user_file_input
# )
