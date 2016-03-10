
import json

obj = json.load(open('10'))


list_obj = []

for key in obj["virustotal"]["scans"]:
	list_obj.append(".".join(["virustotal", "scans", str(key), "result"]) )

print list_obj