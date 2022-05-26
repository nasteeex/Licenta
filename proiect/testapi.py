import requests
import json

urlLocation = "http://localhost:8000/api/location/"
urlCompany = "http://localhost:8000/api/company/"

payloadLocation = {}
headersLocation = {}
payloadCompany = {}
headersCompany = {}

responseLocation = requests.request("GET", urlLocation, headers=headersLocation, data=payloadLocation)
responseCompany = requests.request("GET", urlCompany, headers=headersCompany, data=payloadCompany)

# payloadAddLocation = json.dumps({
#   "city": "Baia Mare1",
#   "country": "Romania"
# })
# headersAddLocation = {
#   'Content-Type': 'application/json'
# }

# responseAddLocation = requests.request("POST", urlLocation, headers=headersAddLocation, data=payloadAddLocation)

# urlMod = "https://localhost:8000/api/location/12/"

# payloadMod = json.dumps({
#   "city": "Arad1",
#   "country": "Romania"
# })
# headersMod = {
#   'Content-Type': 'application/json'
# }

# responseMod = requests.request("PUT", urlMod, headers=headersMod, data=payloadMod)

# print(responseMod.text)
# print()
print(responseLocation.text)
print()
print(responseCompany.text)
# print()
# print(responseAddLocation.text)
