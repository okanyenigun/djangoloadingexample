import time
import json
from .service import Regression
from django.shortcuts import render,redirect
from django.views import View
from django.http import HttpResponse
# Create your views here.

class MainView(View):
    
    def get(self,request):
        if "submit" in request.GET:
            return redirect("result")
        return render(request,'./templates/index.html')


class ResultView(View):
    
    def get(self,request):
        return render(request,'./templates/results.html')

    
    @staticmethod
    def ajax_view(request):
        start_time = time.time()
        R = Regression()
        rmse = R.get_rmse()
        context = {"rmse":rmse,"time":time.time()-start_time}
        data = json.dumps(context)
        return HttpResponse(data,content_type="application/json")