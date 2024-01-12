from django.shortcuts import render, HttpResponse
#from chemistry.visual_script import upload_csv, parse_data, visualize_molecules
# View is where you allow for contents to be seen
# Create your views here.

# below is a func/method , it "renders" the html
#def home(request):
    #return render(request, "home.html")



#uploaded_data = upload_csv('file_path')
#then
#parsed_data = parse_data(uploaded_data) 
#visualize_molecules = visualize_molecules(parsed_data) #this is a picture file


def step1(request):
    return render(request, "website/step1.html")
