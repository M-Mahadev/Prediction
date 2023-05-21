from django.shortcuts import render

import pickle
import os
import sklearn
import numpy

def home(req):
    return render(req, 'index.html')

def result(req):
    feilds=['age','gender','cp','trestbps','cholestrol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    vals=[]
    for f in feilds:
        vals.append(req.POST.get(f))
    v = numpy.array(vals).reshape(1,-1)
    model_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(v)
    if predictions==1:
        return render(req, 'result.html', context={'disease':'True'})
    elif predictions==0:
        return render(req, 'result.html', context={'disease':'False'})
    return render(req,'result.html')