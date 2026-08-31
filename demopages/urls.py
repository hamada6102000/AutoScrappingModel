from django.urls import path

from . import views

app_name = "demopages"

urlpatterns = [
    path("product/", views.product, name="product"),
    path("product/<str:version>/", views.product_version, name="product_version"),
]
