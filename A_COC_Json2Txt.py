import json

with open("E:\\COCO\\annotations_trainval2014\\annotations\\instances_train2014.json",mode="r") as rh:
    line = rh.readline()
    data = json.loads(line)


for k in data.keys():
    print(k)
images = data["images"]
print(type(images))
print(len(images))
### it is a dict : iscrowd, image_id, bbox, category_id
annotations = data["annotations"]
print(type(annotations))
print(len(annotations))
### it is a dict : id, name
categories = data["categories"]
print(type(categories))
print(len(categories))
for c in categories:
    print(c)

with open("./imageID2fileNameTrain.txt","w") as wh:
    for image in images:
        wh.write(str(image["id"]) + "\t" + image["file_name"] + "\n")
with open("./imageID2infoTrain.txt","w") as wh:
    for ann in annotations:
        bbox = ann["bbox"]
        minX = bbox[0]
        minY = bbox[1]
        maxX = minX + bbox[2]
        maxY = minY + bbox[3]
        wh.write(str(ann["image_id"]) + "\t" + str(minX) + "," + str(minY) + "," + str(maxX) + "," + str(maxY)
                 + "\t" + str(ann["category_id"]) + "\t" + str(ann["iscrowd"]) + "\n")




