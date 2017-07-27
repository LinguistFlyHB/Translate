# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.shortcuts import render_to_response
from . import translate as t
import json


# def execute(request):
#     return render_to_response('../templates/execute.html')


def translate(request):
    result = ''
    request.encoding = 'utf-8'
    if 'raw' in request.GET:
        message = request.GET['raw']
        result = t.decode(str(message))
        # result = message
    return render_to_response('../templates/execute.html', {'english': message, 'french': result})