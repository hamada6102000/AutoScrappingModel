from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.target_list, name="target_list"),
    path("targets/<int:pk>/", views.target_detail, name="target_detail"),
    path("targets/<int:pk>/run/", views.run_target, name="run_target"),
    path("runs/<int:pk>/", views.run_detail, name="run_detail"),
    path("changes/", views.change_event_log, name="change_events"),
    path("simulate/", views.simulate_change, name="simulate_change"),
]
