# Overview

In this repository we provide the code for a **L**ow-**C**ode platform for **d**ata-science and **s**treaming, **LODS**. This platform enables users without any IT-background to solve data-science and streaming-problems.

To start, run app.py. Relevant scripts are in folders **lib**, **static** and **templates**.

# Technical stuff

- Environment variables needed. Saved locally in .env. Check variables in powershell:
```Write-Host "The value of MY_VAR is: $($env:openai__api__key)"```
- session size limited to 4kb --> save files locally

# Docker
- Important to remove windows dependencies from requirements (e.g. pywin) when building on linux.
- Problems with version of tensorflow --> remove specific version
- run commands:
```
docker build -t my-flask-app .
docker run -it my-flask-app /bin/bash
docker run -p 5000:5000 my-flask-app
```
- known errors:
   - [https://stackoverflow.com/questions/42787779/docker-login-error-storing-credentials-write-permissions-error](https://stackoverflow.com/questions/42787779/docker-login-error-storing-credentials-write-permissions-error) solution: delete config-file in C:\Users\oliver.koehn\.docker

- **IMPORTANT TO USE LOWERCASES FOR REGISTRY NAME**
- Admin account needs be [enabled](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli):

```az acr update -n flaskdummyappdatascience --admin-enabled true```

- logs: ```az webapp log download --resource-group rg-lob-adv-cc-dai --name azuredsoliver```
- push docker-image to registry [https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli?tabs=azure-cli](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli?tabs=azure-cli)
   - needs to be renamed with path ro repo and then provide a path, e.g. samples/flaskdummy

   - example:
```
docker pull nginx
docker run -it --rm -p 8080:80 nginx
docker tag nginx flaskdummyappdatascience.azurecr.io/samples/nginx
az login
az acr login -n flaskdummyappdatascience
docker push flaskdummyappdatascience.azurecr.io/nginx
```

```
C:\WINDOWS\system32> docker tag nginx flaskdummyappdatascience.azurecr.io/samples/nginx
C:\WINDOWS\system32> az login
A web browser has been opened at https://login.microsoftonline.com/organizations/oauth2/v2.0/authorize. Please continue the login in the web browser. If no web browser is available or if the web browser fails to open, use device code flow with `az login --use-device-code`.
[
  {
    "cloudName": "AzureCloud",
    "homeTenantId": "xxx",
    "id": "xxx",
    "isDefault": true,
    "managedByTenants": [],
    "name": "xxx",
    "state": "Enabled",
    "tenantId": "xxx",
    "user": {
      "name": "xxx",
      "type": "user"
    }
  }
]
C:\WINDOWS\system32> az acr login -n flaskdummyappdatascience
Login Succeeded
C:\WINDOWS\system32> docker push flaskdummyappdatascience.azurecr.io/nginx
Using default tag: latest
The push refers to repository [flaskdummyappdatascience.azurecr.io/nginx]
61a7fb4dabcd: Layer already exists
bcc6856722b7: Layer already exists
188d128a188c: Layer already exists
7d52a4114c36: Layer already exists
3137f8f0c641: Layer already exists
84619992a45b: Layer already exists
ceb365432eec: Layer already exists
latest: digest: sha256:678226242061e7dd8c007c32a060b7695318f4571096cbeff81f84e50787f581 size: 1778
C:\WINDOWS\system32>
```

# Azure for PS

- Update powershell:

```winget install --id Microsoft.Powershell --source winget```

- [Install Azure for Powershell](https://learn.microsoft.com/en-us/powershell/azure/install-azps-windows?view=azps-11.3.0&tabs=powershell&pivots=windows-psgallery):

```Install-Module -Name Az -Repository PSGallery -Force```

# Types

## Regression/Classification

- **Group-expressions**: If there are multiple tables, it is possible that new columns are added as grouping. For multiple tables we provide ID only, so need to care for calculation.
- **add new column**: In contrast to grouping, new columns combine two or more columns. Therefore we need to extend teh user input by these newly calculated columns. 

# Techniques:

- **Session**: The session object saves user-specific content. Its saved server-side, but connected to the user via a token. So user information of a session can be stored. maybe problems with size, see here [https://stackoverflow.com/questions/44013058/flask-session-max-size-too-small](https://stackoverflow.com/questions/44013058/flask-session-max-size-too-small)
- **render or redirect**: "If you want the user to see the result immediately on the same page without a full page reload, using render_template is a more seamless approach. It provides a better user experience by updating the content dynamically on the client side. Use redirect if you want to send the user to a completely new page for the result."
- **Routing**. Soemtimes we need verify information. Therefore we provide a button for checking. To go to other page we need provide another button immediately refering to other page.

- **Reloading moduoles in jupyter**: Jupyter typically does not reload moduoles. To force reload use this ([https://saturncloud.io/blog/jupyter-notebook-reload-module-a-comprehensive-guide/](https://saturncloud.io/blog/jupyter-notebook-reload-module-a-comprehensive-guide/)):

```
%load_ext autoreload
%autoreload 2
```

- **Detect modules**: Make jupyter detect modules in different path:

```
import sys
# add to path
sys.path.append(r"C:\Users\oliver.koehn\Documents\projects\masterThesisLODS\app")
sys.path.append(r"C:\Users\oliver.koehn\Documents\projects\masterThesisLODS\app\lib")
```



# Examples

## Regression

- autoMpg:
   - Data from: [https://archive.ics.uci.edu/dataset/9/auto+mpg](https://archive.ics.uci.edu/dataset/9/auto+mpg)
   - Example: [https://www.tensorflow.org/tutorials/keras/regression](https://www.tensorflow.org/tutorials/keras/regression)

- real estate:
   - Data from: [https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction](https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction)

- medical cost:
   - Data from: [https://www.kaggle.com/datasets/mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Classification

- berka:
   - Data from: [https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions](https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions)

- income:
   - data from: [https://archive.ics.uci.edu/dataset/20/census+income](https://archive.ics.uci.edu/dataset/20/census+income)

- star classification:
   - data from: [https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv](https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Stars.csv)
   - evaluation: [https://www.kaggle.com/code/ybifoundation/stars-classification/notebook](https://www.kaggle.com/code/ybifoundation/stars-classification/notebook)

