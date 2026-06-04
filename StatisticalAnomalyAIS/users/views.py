# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel, TokenCountModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from datetime import datetime, timedelta
from jose import JWTError, jwt
import numpy as np
import os
import os

SECRET_KEY = "ce9941882f6e044f9809bcee90a2992b4d9d9c21235ab7c537ad56517050f26b"
ALGORITHM = "HS256"

import socket


def get_ipv4_address():
    try:
        # connect to an external host, doesn't send data
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google DNS
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return f"Error: {e}"


def create_access_token(data: dict):
    to_encode = data.copy()
    # expire time of the token
    expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # return the generated token
    return encoded_jwt


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HttpResponse(
            status_code=HttpResponse(status=204),
            detail="Could not validate credentials",
        )


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            loginId = form.cleaned_data['loginid']
            TokenCountModel.objects.create(loginid=loginId, count=0)
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}
                token_jwt = create_access_token(data)
                request.session['token'] = token_jwt
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {'ip': get_ipv4_address()})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {'ip': get_ipv4_address()})


def viewAISDataset(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'AIS.csv')
    df = pd.read_csv(path)
    df = df.to_html(index=False)
    return render(request, 'users/viewdataset.html', {'data': df})


def timingAnalysis(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'AIS.csv')
    df = pd.read_csv(path)
    from .utility import build_anomaly_detection
    timingAn = build_anomaly_detection.analyze_timing(df)

    return render(request, 'users/timingAnalysis.html', {'data': timingAn.to_html})


def correlationVrepVcalc(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'AIS.csv')
    df = pd.read_csv(path)
    from .utility import build_anomaly_detection
    corr = build_anomaly_detection.analyze_correlation(df)
    corr = corr.to_frame()

    return render(request, 'users/corrVrepVcalc.html', {'data': corr.to_html})


def anomalyDetection(request):
    from django.conf import settings
    import pandas as pd
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'AIS.csv')
    df = pd.read_csv(path)
    from .utility import build_anomaly_detection
    df = build_anomaly_detection.calculate_features(df)
    df_anomalies = build_anomaly_detection.detect_anomalies(df)

    return render(request, 'users/viewAnomalies.html', {'data': df_anomalies.to_html(index=False)})


def selfTest(request):
    if request.method == 'POST':
        mmsi = int(request.POST.get('mmsi'))
        lat = float(request.POST.get('lat'))
        lon = float(request.POST.get('lon'))
        time1 = int(request.POST.get('time'))
        speed = float(request.POST.get('speed'))
        from .utility import testInput
        var1, var2, var3, msg, threat = testInput.start_test(mmsi, lat, lon, time1, speed)
        return render(request, 'users/selfassessment.html',
               {'var1': var1, 'var2': var2, 'var3': var3, 'msg': msg, 'threat': threat})


    else:
        return render(request, 'users/selfassessment.html', {})
