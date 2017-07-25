# -*- coding: utf-8 -*-

from django.http import HttpResponse
from django.shortcuts import render_to_response


# 表单
def execute(request):
    return render_to_response('../templates/execute.html')


# 接收原始语言
def translate(request):
    request.encoding = 'utf-8'
    if 'raw' in request.GET:
        message = request.GET['raw']
    else:
        message = '请输入待翻译语句'
    return HttpResponse(message)