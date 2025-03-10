
from django.urls import path, include
# 默认在views中进行编写视图函数
from .views import *
from django.conf.urls.static import static

urlpatterns = [
    # 本应用下的路由，第一个参数是路由名，第二个参数是视图函数名，也可以有第三个参数，
    # 也就是命名空间,name=index(一般三个都与视图函数名一致)
    path("predict/", predict),
    path("evaluate/", evaluate, name="evaluate"),
    path('predict/',upload_file, name='predict'),
]
urlpatterns += static('/upload/', document_root=settings.MEDIA_ROOT)  #加上这一行
