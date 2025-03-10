import os

# 先导入HttpResponse进行测试
from django.shortcuts import render,HttpResponse
import subprocess
from django.conf import settings
from .models import UploadFileForm
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from .models import UploadFilesForm2
from django.http import FileResponse


# Create your views here.
def index(request):
    # 这里返回使用HttpResponse进行测试（只能输出一句话）
    return HttpResponse("欢迎来到首页")

##预测代码
def predict2(valid_path_csv, your_out_dir):
    # 外部参数
    arg1 = valid_path_csv
    arg2 = your_out_dir
    
    # 要执行的 Python 脚本文件
    script_file = "C:/Users/30961/Desktop/test/newtest/django/mysite/src/predict.py"
    subprocess.run(["python", script_file, arg1, arg2])

def evaluate2(valid_path_csv, preds_csv):
    evaluate = "C:/Users/30961/Desktop/test/newtest/django/mysite/src/evaluate.py"
    # 使用 subprocess 模块执行另一个 Python 脚本，并传入外部参数
    arg1 = valid_path_csv
    arg2 = preds_csv
    arg3 = "C:/Users/30961/Desktop/test/newtest/django/mysite/MRNet-v1.0/valid_labels.csv"
    subprocess.run(["python", evaluate, arg1, arg2, arg3])

def upload_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename)
        predict(uploaded_file_path,)
    return render(request, 'upload.html')




def predict(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_path = os.path.join(settings.MEDIA_ROOT, filename).replace('\\', '/')
        
        # 获取用户选择的输出路径，并确保使用 "/" 作为路径分隔符
        output_path = request.POST.get('output_path').replace('\\', '/')
        predict2(uploaded_file_path, output_path)
        # 在这里执行你的文件处理逻辑
        # 例如，将上传的文件复制到指定的输出路径
        return HttpResponse("预测出的csv文件已经在你输入的文件路径中")
    
    return render(request, 'predict.html')


def evaluate(request):
    if request.method == 'POST':
        file1 = request.FILES['file1']
        file2 = request.FILES['file2']
        fs = FileSystemStorage()
        filename1 = fs.save(file1.name, file1)
        filename2 = fs.save(file2.name, file2)
        # 保存上传的文件到 MEDIA_ROOT 目录中
        file1_path = os.path.join(settings.MEDIA_ROOT, filename1).replace('\\', '/')
        file2_path = os.path.join(settings.MEDIA_ROOT, filename2).replace('\\', '/')

        # 构造文件的绝对地址并返回
        file1_absolute_url = request.build_absolute_uri(file1_path)
        file2_absolute_url = request.build_absolute_uri(file2_path)
        evaluate2(file1_absolute_url, file2_absolute_url)


        plot_path = os.path.join(settings.MEDIA_ROOT,'C:/Users/30961/Desktop/test/newtest/django/mysite/media/auc.jpg')
        image_path = 'C:/Users/30961/Desktop/test/newtest/django/mysite/media/auc.jpg'
        absolute_image_path = os.path.join(settings.BASE_DIR, 'C:/Users/30961/Desktop/test/newtest/django/mysite/media/image/auc.jpg')

    # 将绝对路径传递给模板
        return render(request, 'show_plot.html', {'absolute_image_path': absolute_image_path})

    return render(request, 'upload_files.html')


