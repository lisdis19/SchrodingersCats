from django.shortcuts import render, HttpResponse
from chemistry.jupyterbackend import process_csv
from chemistry.backendUser import process_molecular_data #describe
# View is where you allow for contents to be seen
# Create your views here.
#from apply_predictive_model import backendUser
# below is a func/method , it "renders" the html
#def home(request):
    #return render(request, "home.html")


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

def contact(request):
    return render(request, "contact.html")

def process_csvFile(request):
    input = request.POST.get("csvFile")
    output = process_csv(input)
    return render(request,"step2.html",{"result":results})

def select_algo(request):
    #input = request.POST.get("choicealgo") #this is the algo value : describe, visulaize...
    input = "describe" #this is the algo value : describe, visulaize...
    print(input)
    if input == "describe":
        output = process_molecular_data("hiv_dataset_3.csv", test_size=0.2, random_state=42)
    print(output)
    if input == "predict":
        output = 
    return render(request, "step3.html",{"algoresult":output})
# upload_csv(
#     #user_file_input
# )
